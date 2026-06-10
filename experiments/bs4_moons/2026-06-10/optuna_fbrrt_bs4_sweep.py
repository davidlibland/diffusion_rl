#!/usr/bin/env python3
"""Optuna re-tune of the FBRRT family on the moons BS=4 benchmark (fixed code).

The original BS=4 study (../2026-05-28/) found FBRRT to be the weakest and
slowest family (best: `fbrrt_td_lambda` plateau -10.71, behind even tuned
off-policy at -9.69).  Those results were produced with FBRRT estimators that
have since been fixed:

  - (B)  local-entropy weights moved OFF the bootstrap target (uniform child
         mean) and onto the regression loss, plumbed through the dataset;
  - (D)  ancestor-aligned TD(lambda) multi-step + multinomial resampling;
  - (A,C) corrected control-variate driver + 1/sqrt(2a) Malliavin scaling;
  - right-endpoint BSDE driver (child-time gradients at t_{i+1}, so targets
         bootstrap from the better-trained side of the network) with the
         drift/driver autograd fused to one child-batch call per step;
  - `fbrrt_cv` through the dataset now uses smc_value as the FROZEN policy
         net (we pass the EMA shadow), so the residual control variate is
         actually exercised (previously v_policy == v_target made it
         identical to plain `fbrrt`).

This script re-tunes the family under the fixed estimators with the same
task / objective / budgets as the rest of the BS=4 study, mirroring the
`ancestral_mc_td_lambda` post-fix re-tune (../2026-05-28/optuna_amctl_bs4_sweep.py):

Phase 1 sweep   : OPT_N_TRIALS trials, TPE + Hyperband, OPT_MAX_STEPS steps.
Phase 2 confirm : top-OPT_TOPK trials (plus best per method, if absent)
                  x OPT_N_SEEDS seeds x OPT_MAX_STEPS steps.
Phase 3 converge: best confirmed config -> OPT_CONV_STEPS steps, serialized.

Design space
------------
method ∈ {fbrrt, fbrrt_td_lambda, fbrrt_cv}   (fbrrt_mc_z excluded: it is
    numerically divergent by construction -- see its docstring)
branch 2..16 (the old sweep's winner sat at the old cap of 10),
mc_samples (=n_particles) 2..32 log, n_steps 10..60, alpha 0..1.5,
entropy_lambda ∈ {inf} ∪ logU[0.05, 5]  (inf = uniform resampling + unweighted
    regression; finite values now reweight the LOSS, not the target),
lambda_eff (fbrrt_td_lambda only), ema_decay (fbrrt_cv policy EMA only),
off_policy_frac 0..0.5, lr logU[1e-4, 3e-3], grad_decay toggled logU[1e-5,1e-1],
loss_type ∈ {quad, mse}.

Objective: detrended-SEM LCB over the last OPT_LCB_TAIL of OPT_N_VAL validation
checkpoints -- identical to the archived study, so numbers are comparable.

Parallel execution (this box: 32-core 9950X + RTX 3090 Ti; one BS=4 run uses
~2 GB VRAM and ~1 core, so we run many at once)
---------------------------------------------------------------------------
The study lives in an optuna JournalStorage (multi-process safe).  Stages are
orchestrated by run_parallel.sh:

  stage 1  K sweep workers      OPT_EXIT_AFTER=sweep OPT_SAMPLER_SEED=42+i
           (global budget enforced via MaxTrialsCallback; TPE constant_liar)
  stage 2  K confirm workers    OPT_EXIT_AFTER=confirm OPT_WORKER_ID=i
           OPT_N_WORKERS=K  (work split by flat (config, seed) index; a
           (config, seed) run is SKIPPED if its metrics.csv is complete, so
           the final pass just aggregates)
  stage 3  final pass           converges the best confirmed config of EACH
           method in parallel (subprocesses with OPT_CONV_TRIAL=<n>), then
           aggregates, plots, and writes the results JSON.

Smoke test:
    OPT_N_TRIALS=2 OPT_MAX_STEPS=200 OPT_N_VAL=10 OPT_LCB_TAIL=5 OPT_TOPK=1 \
    OPT_N_SEEDS=2 OPT_CONV_STEPS=300 \
    python experiments/bs4_moons/2026-06-10/optuna_fbrrt_bs4_sweep.py
"""

import gc
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from einops import reduce
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork


# ── Device ──────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── Shared setup (identical task to the rest of the BS=4 study) ─────────────
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scaler = StandardScaler(); X = scaler.fit_transform(X)
clf = GaussianMixture(n_components=100, covariance_type="spherical"); clf.fit(X)
_means = torch.from_numpy(clf.means_).double()
_sigma2 = torch.from_numpy(clf.covariances_).double()
_weights = torch.from_numpy(clf.weights_).double()
_sigmas = torch.sqrt(_sigma2)[:, None]; _weights_col = _weights[:, None]

D = 2; a = 1.0; c = torch.tensor([1.0, 0.0])
means_np = clf.means_; sigmas_np = np.sqrt(clf.covariances_); weights_np = clf.weights_


def gmm_drift(xt, ts, a_):
    ts = ts.reshape(-1, 1); xt_ = xt[..., None]
    means_ = _means.float().to(xt).T[None, ...]; ts_ = ts[..., None]
    sigmas_ = _sigmas.float().to(xt).T; weights_ = _weights_col.float().to(xt).T
    denom = 2 * a_ * (1 - ts) + ts * sigmas_**2
    le = -reduce((xt_ - means_ * ts_) ** 2, "n d m -> n m", "sum") / (2 * ts * denom)
    lsf = torch.log(2 * a_ * (1 - ts) / denom) * D / 2
    lrw = torch.log(weights_) + le + lsf
    lw = lrw - torch.logsumexp(lrw, dim=1, keepdim=True)
    lw = torch.where((ts == 0), torch.log(weights_), lw)
    nm = (2 * a_ * (1 - ts_) * means_ + xt_ * sigmas_[None, ...] ** 2) / denom[:, None, :]
    us = (nm - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(torch.exp(lw)[:, None, :] * us, "n d m -> n d", "sum")


def base_drift(x, t):
    return gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)


def reward_fn(x):
    return -10 * (x - c.to(x)).square().sum(dim=1)


def gmm_sample(n):
    k_ = np.random.choice(len(weights_np), size=n, p=weights_np)
    return means_np[k_] + sigmas_np[k_, np.newaxis] * np.random.randn(n, D)


class OnPolicyValueLive(OnPolicyValue):
    """Inference/drift use the LIVE network (no EMA in drift) — the BS=4 default."""

    def drift(self, x, t, beta=1, use_ema=False):
        return super().drift(x, t, beta=beta, use_ema=use_ema)


# ── Constants ──────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = "experiments/bs4_moons/2026-05-28"
_TAG = os.environ.get("OPT_LOGTAG", "")  # suffix for smoke-test isolation
LOG_DIR = f"lightning_logs/optuna_fbrrt2{_TAG}"
CONFIRM_DIR = f"lightning_logs/optuna_fbrrt2{_TAG}_confirm"
CONV_LOG_DIR = f"lightning_logs/optuna_fbrrt2{_TAG}_converge"
CKPT_DIR = f"checkpoints/optuna_fbrrt2{_TAG}_converge"
STUDY_JOURNAL = f"{HERE}/optuna_fbrrt2_journal.log"
STUDY_NAME = "fbrrt_bs4_lcb_fixed_v1"

METHODS = ["fbrrt", "fbrrt_td_lambda", "fbrrt_cv"]

BS = 4
DS_BATCH = 64
MAX_STEPS = int(os.environ.get("OPT_MAX_STEPS", 5000))
N_VAL = int(os.environ.get("OPT_N_VAL", 50))
LCB_TAIL = int(os.environ.get("OPT_LCB_TAIL", 20))
LCB_Z = 1.645
N_TRIALS = int(os.environ.get("OPT_N_TRIALS", 80))
TOPK = int(os.environ.get("OPT_TOPK", 3))
N_SEEDS = int(os.environ.get("OPT_N_SEEDS", 5))
CONV_STEPS = int(os.environ.get("OPT_CONV_STEPS", 50000))
CONV_VAL_EVERY = int(os.environ.get("OPT_CONV_VAL_EVERY", 1000))

# Parallel-execution knobs (see module docstring).
EXIT_AFTER = os.environ.get("OPT_EXIT_AFTER", "")          # "", "sweep", "confirm"
WORKER_ID = int(os.environ.get("OPT_WORKER_ID", 0))
N_WORKERS = int(os.environ.get("OPT_N_WORKERS", 1))
SAMPLER_SEED = int(os.environ.get("OPT_SAMPLER_SEED", 42))
CONV_TRIAL = os.environ.get("OPT_CONV_TRIAL")              # converge-only mode

all_rewards = reward_fn(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()
val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


class OptunaPruning(Callback):
    def __init__(self, trial, monitor="val_reward_mean"):
        super().__init__(); self.trial = trial; self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        m = trainer.callback_metrics.get(self.monitor)
        if m is None:
            return
        step = int(trainer.global_step)
        self.trial.report(float(m), step)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"pruned at step {step}")


def read_curve(csv_path):
    if not os.path.exists(csv_path):
        return np.array([]), np.array([])
    df = pd.read_csv(csv_path)
    if "val_reward_mean" not in df.columns:
        return np.array([]), np.array([])
    sub = df.dropna(subset=["val_reward_mean"])
    return sub["step"].to_numpy(), sub["val_reward_mean"].to_numpy()


def lcb_of(curve):
    """Detrended-SEM lower confidence bound over the last LCB_TAIL points."""
    if len(curve) < 5:
        return -100.0
    tail = curve[-LCB_TAIL:]
    n = len(tail)
    xx = np.arange(n, dtype=float)
    A = np.vstack([xx, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, tail, rcond=None)
    resid = tail - A @ coef
    sigma = float(np.sqrt((resid ** 2).sum() / max(1, n - 2)))
    return float(tail.mean() - LCB_Z * sigma / np.sqrt(n))


# ── Hyperparameter design space ────────────────────────────────────────────
def sample_params(trial):
    p = {
        "method": trial.suggest_categorical("method", METHODS),
        "n_steps": trial.suggest_int("n_steps", 10, 60),
        "mc_samples": trial.suggest_int("mc_samples", 2, 32, log=True),
        "branch": trial.suggest_int("branch", 2, 16),
        "alpha": trial.suggest_float("alpha", 0.0, 1.5),
        "off_policy_frac": trial.suggest_float("off_policy_frac", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "loss_type": trial.suggest_categorical("loss_type", ["quad", "mse"]),
    }
    # entropy_lambda: inf = uniform resampling + unweighted regression;
    # finite values tilt the forward resampling AND weight the loss.
    if trial.suggest_categorical("ent_inf", [True, False]):
        p["entropy_lambda"] = float("inf")
    else:
        p["entropy_lambda"] = trial.suggest_float(
            "entropy_lambda", 0.05, 5.0, log=True)
    if p["method"] == "fbrrt_td_lambda":
        p["lambda_eff"] = trial.suggest_float("lambda_eff", 0.0, 1.0)
    if p["method"] == "fbrrt_cv":
        # decay of the EMA shadow used as the frozen v_policy
        p["ema_decay"] = trial.suggest_float("ema_decay", 0.90, 0.999)
    if trial.suggest_categorical("use_grad_decay", [True, False]):
        p["grad_decay"] = trial.suggest_float("grad_decay", 1e-5, 1e-1, log=True)
        p["use_grad_decay"] = True
    else:
        p["use_grad_decay"] = False
    return p


def trial_params(t):
    """Reconstruct the full params dict from a finished trial."""
    pr = dict(t.params)
    out = {
        "method": pr["method"], "n_steps": pr["n_steps"],
        "mc_samples": pr["mc_samples"], "branch": pr["branch"],
        "alpha": pr["alpha"], "off_policy_frac": pr["off_policy_frac"],
        "lr": pr["lr"], "loss_type": pr["loss_type"],
        "use_grad_decay": pr.get("use_grad_decay", False),
    }
    out["entropy_lambda"] = (float("inf") if pr.get("ent_inf", False)
                             else pr["entropy_lambda"])
    if out["method"] == "fbrrt_td_lambda":
        out["lambda_eff"] = pr["lambda_eff"]
    if out["method"] == "fbrrt_cv":
        out["ema_decay"] = pr["ema_decay"]
    if out["use_grad_decay"]:
        out["grad_decay"] = pr["grad_decay"]
    return out


def build(params, seed):
    L.seed_everything(seed, workers=True)
    grad_decay = params["grad_decay"] if params.get("use_grad_decay") else None
    ema_decay = params.get("ema_decay", 0.99)
    vm = ValueNetwork(D, bias=bias_val)
    model = OnPolicyValueLive(
        base_score_module=base_drift, value_module=vm, reward_function=reward_fn,
        dim=D, a=a, lr=params["lr"], loss_type=params["loss_type"],
        grad_decay=grad_decay, ema_decay=ema_decay,
    ).to(DEVICE)
    # smc_value is unused by fbrrt / fbrrt_td_lambda; for fbrrt_cv it is the
    # FROZEN policy net (EMA shadow) defining the control + RCV anchor.
    smc_value = model.ema if params["method"] == "fbrrt_cv" else model.value_module
    ds = OnPolicySMCDataset(
        dim=D, drift=base_drift, value=model.value_module,
        smc_value=smc_value, reward=reward_fn, device=DEVICE, a=a,
        batch_size=DS_BATCH, n_steps=params["n_steps"],
        mc_samples_per_step=params["mc_samples"],
        sampling_method=params["method"],
        branch=params["branch"], entropy_lambda=params["entropy_lambda"],
        fbrrt_alpha=params["alpha"],
        lambda_eff=params.get("lambda_eff", 0.5),
        off_policy_frac=params["off_policy_frac"], include_t_zero=False,
        random_t=False, generating_function=gmm_sample,
    )
    loader = DataLoader(ds, batch_size=BS)
    return model, vm, ds, loader


def fmt(p):
    s = (f"{p['method']} mc={p['mc_samples']} br={p['branch']} "
         f"ns={p['n_steps']} a={p['alpha']:.2f} "
         f"ent={'inf' if p['entropy_lambda'] == float('inf') else format(p['entropy_lambda'], '.2f')} "
         f"ofp={p['off_policy_frac']:.2f} lr={p['lr']:.1e} loss={p['loss_type']} "
         f"{'gd='+format(p['grad_decay'],'.1e') if p.get('use_grad_decay') else 'gd=off'}")
    if "lambda_eff" in p:
        s += f" lam={p['lambda_eff']:.2f}"
    if "ema_decay" in p:
        s += f" ema={p['ema_decay']:.3f}"
    return s


# ── Phase 1: sweep ─────────────────────────────────────────────────────────
def objective(trial):
    p = sample_params(trial)
    name = f"trial_{trial.number:04d}"
    csv = f"{LOG_DIR}/{name}/version_0/metrics.csv"
    t0 = time.time()
    model, vm, ds, loader = build(p, 1234 + trial.number)
    logger = CSVLogger(LOG_DIR, name=name, version=0)
    trainer = L.Trainer(
        max_steps=MAX_STEPS, val_check_interval=max(1, MAX_STEPS // N_VAL),
        callbacks=[OptunaPruning(trial)], logger=logger,
        enable_checkpointing=False, enable_progress_bar=False)
    try:
        trainer.fit(model, loader, val_dataloaders=val_loader)
    except optuna.TrialPruned:
        print(f"  {name}: pruned ({(time.time()-t0)/60:.1f} min)", flush=True)
        raise
    except (RuntimeError, ValueError) as e:
        del model, vm, trainer, loader, ds
        gc.collect(); empty_cache()
        print(f"  {name}: error {type(e).__name__}: {str(e)[:120]} -> -100  "
              f"[{fmt(p)}]", flush=True)
        return -100.0
    st, cv = read_curve(csv)
    del model, vm, trainer, loader, ds
    gc.collect(); empty_cache()
    lcb = lcb_of(cv)
    print(f"  {name}: LCB={lcb:.3f}  [{fmt(p)}]  {(time.time()-t0)/60:.1f} min",
          flush=True)
    return lcb


t_start = time.time()
sampler = TPESampler(multivariate=True, group=True, seed=SAMPLER_SEED,
                     constant_liar=True)  # constant_liar: parallel-safe TPE
pruner = HyperbandPruner(min_resource=500, max_resource=MAX_STEPS,
                         reduction_factor=3)
storage = JournalStorage(JournalFileBackend(STUDY_JOURNAL))
study = optuna.create_study(
    study_name=STUDY_NAME, storage=storage, load_if_exists=True,
    direction="maximize", sampler=sampler, pruner=pruner)
n_done = len([t for t in study.trials if t.state.is_finished()])
print(f"PHASE 1 — FBRRT (fixed) sweep  device={DEVICE}  worker={WORKER_ID}  "
      f"done={n_done} remaining={max(0, N_TRIALS-n_done)}", flush=True)


def _cb(study, trial):
    done = len([t for t in study.trials if t.state.is_finished()])
    try:
        best = study.best_value
    except ValueError:
        best = float("nan")
    print(f"[{done}/{N_TRIALS}] elapsed={(time.time()-t_start)/60:.1f} min "
          f"| best LCB={best:.3f}", flush=True)


if n_done < N_TRIALS:
    # Global budget across all workers: MaxTrialsCallback counts finished
    # trials in the shared storage and stops this worker when the budget is
    # met (slight overshoot by in-flight trials is fine).
    study.optimize(
        objective, n_trials=None,
        callbacks=[_cb, MaxTrialsCallback(
            N_TRIALS, states=(TrialState.COMPLETE, TrialState.PRUNED))],
        gc_after_trial=True)

if EXIT_AFTER == "sweep":
    print(f"[worker {WORKER_ID}] sweep stage done -- exiting.", flush=True)
    raise SystemExit(0)

comp = [t for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None]
comp.sort(key=lambda t: t.value, reverse=True)
chosen = comp[:TOPK]
# Ensure the best trial of each method is represented (like the family rule
# in the original "other" pipeline).
have = {trial_params(t)["method"] for t in chosen}
for m in METHODS:
    if m not in have:
        best_m = next((t for t in comp if trial_params(t)["method"] == m), None)
        if best_m is not None and best_m.value > -50:
            chosen.append(best_m)
            have.add(m)

print("\n" + "=" * 80)
print(f"Selected for {N_SEEDS}-seed confirm:")
for t in chosen:
    print(f"  trial {t.number:>3}  LCB={t.value:>8.3f}  [{fmt(trial_params(t))}]")
print("=" * 80, flush=True)
if EXIT_AFTER != "confirm" and CONV_TRIAL is None:  # don't race on the json
    json.dump([{"trial": t.number, "lcb": t.value, "params": trial_params(t)}
               for t in chosen],
              open(f"{HERE}/optuna_fbrrt2_top.json", "w"), indent=2, default=str)


# ── Phase 2: confirm ───────────────────────────────────────────────────────
# Idempotent: a (config, seed) run whose metrics.csv is already complete is
# only READ, never retrained -- so K parallel confirm workers (partitioned by
# the flat run index) populate the CSVs, and the final pass just aggregates.
print(f"\nPHASE 2 — confirm × {N_SEEDS} seeds × {MAX_STEPS} steps "
      f"(worker {WORKER_ID}/{N_WORKERS})\n", flush=True)
MIN_CURVE = max(5, int(N_VAL * 0.6))  # "complete" threshold for a CSV


def _confirm_run(params, trial, s):
    """Train one (config, seed) confirm run unless its CSV is complete."""
    name = f"t{trial}_seed{s:02d}"
    csv = f"{CONFIRM_DIR}/{name}/version_0/metrics.csv"
    _, cv = read_curve(csv)
    if len(cv) >= MIN_CURVE:
        return csv, False  # already done
    for vv in range(3):
        pth = f"{CONFIRM_DIR}/{name}/version_{vv}"
        if os.path.exists(pth):
            shutil.rmtree(pth)
    t0 = time.time()
    try:
        model, vm, ds, loader = build(params, 1000 + s)
        logger = CSVLogger(CONFIRM_DIR, name=name, version=0)
        tr = L.Trainer(
            max_steps=MAX_STEPS, val_check_interval=max(1, MAX_STEPS // N_VAL),
            logger=logger, enable_checkpointing=False,
            enable_progress_bar=False)
        tr.fit(model, loader, val_dataloaders=val_loader)
        del model, vm, tr, loader, ds
    except (RuntimeError, ValueError) as e:
        print(f"  {name}: error {type(e).__name__}: {str(e)[:120]}",
              flush=True)
    gc.collect(); empty_cache()
    print(f"  {name}: trained ({(time.time()-t0)/60:.1f} min)", flush=True)
    return csv, True


run_idx = 0
for t in chosen:
    params = trial_params(t)
    for s in range(N_SEEDS):
        if run_idx % N_WORKERS == WORKER_ID:
            _confirm_run(params, t.number, s)
        run_idx += 1

if EXIT_AFTER == "confirm":
    print(f"[worker {WORKER_ID}] confirm stage done -- exiting.", flush=True)
    raise SystemExit(0)

# Aggregate (CSVs now complete -- either from workers or the loop above).
confirm = {}
for t in chosen:
    params = trial_params(t); trial = t.number
    seed_lcbs, seed_bests = [], []
    for s in range(N_SEEDS):
        csv = f"{CONFIRM_DIR}/t{trial}_seed{s:02d}/version_0/metrics.csv"
        st, cv = read_curve(csv)
        lcb = lcb_of(cv) if len(cv) else -100.0
        best = float(cv.max()) if len(cv) else -100.0
        seed_lcbs.append(lcb); seed_bests.append(best)
        print(f"  t{trial}_seed{s:02d}: LCB={lcb:.3f} best={best:.3f}",
              flush=True)
    seed_lcbs = np.array(seed_lcbs); seed_bests = np.array(seed_bests)
    confirm[trial] = {
        "params": {k: (str(v) if v == float("inf") else v)
                   for k, v in params.items()},
        "_params_raw": params,
        "lcb_mean": float(seed_lcbs.mean()),
        "lcb_sd": float(seed_lcbs.std(ddof=1)) if len(seed_lcbs) > 1 else 0.0,
        "lcb_values": seed_lcbs.tolist(),
        "best_mean": float(seed_bests.mean()),
        "best_sd": float(seed_bests.std(ddof=1)) if len(seed_bests) > 1 else 0.0,
    }
    print(f"  >>> t{trial} ({params['method']}): LCB {seed_lcbs.mean():.3f} ± "
          f"{confirm[trial]['lcb_sd']:.3f}  best {seed_bests.mean():.3f} ± "
          f"{confirm[trial]['best_sd']:.3f}", flush=True)

if CONV_TRIAL is None:  # converge workers don't race on the json
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "_params_raw"}
               for k, v in confirm.items()},
              open(f"{HERE}/optuna_fbrrt2_confirm_results.json", "w"), indent=2)

best_trial = max(confirm, key=lambda tr: confirm[tr]["lcb_mean"])
best_conf = confirm[best_trial]
best_params = best_conf["_params_raw"]
print("\n" + "=" * 80)
print(f"Confirmed winner: trial {best_trial}  "
      f"LCB {best_conf['lcb_mean']:.3f} ± {best_conf['lcb_sd']:.3f}")
print(f"  {fmt(best_params)}")
print("=" * 80, flush=True)


# ── Phase 3: converge best config of EACH method (parallel) ────────────────
def detect_convergence(steps, curve, win=8):
    if len(curve) < win + 4:
        return None, float(curve[-1]) if len(curve) else float("nan")
    sm = pd.Series(curve).rolling(win, min_periods=1).mean().to_numpy()
    tail = sm[-max(4, len(sm) // 5):]
    plateau = float(tail.mean())
    noise = float(np.std(curve[len(curve) // 2:] - sm[len(sm) // 2:]))
    conv_step = None
    for i in range(len(sm)):
        if sm[i] >= plateau - 0.5 * noise and np.all(sm[i:] >= plateau - noise):
            conv_step = int(steps[i]); break
    return conv_step, plateau


def converge_one(trial_no, params):
    """Train one config to CONV_STEPS, serialize, write conv_t{n}.json."""
    tag = f"fbrrt2_t{trial_no}_{params['method']}_converge"
    print(f"=== {tag} ===\n  {fmt(params)}", flush=True)
    for vv in range(3):
        pth = f"{CONV_LOG_DIR}/{tag}/version_{vv}"
        if os.path.exists(pth):
            shutil.rmtree(pth)
    ckdir = f"{CKPT_DIR}/{tag}"
    if os.path.exists(ckdir):
        shutil.rmtree(ckdir)
    os.makedirs(ckdir, exist_ok=True)
    t0 = time.time()
    model, vm, ds, loader = build(params, 20240)
    logger = CSVLogger(CONV_LOG_DIR, name=tag, version=0)
    ckpt = ModelCheckpoint(dirpath=ckdir, save_last=True, save_top_k=1,
                           monitor="val_reward_mean", mode="max",
                           filename="best")
    tr = L.Trainer(
        max_steps=CONV_STEPS, val_check_interval=CONV_VAL_EVERY,
        callbacks=[ckpt], logger=logger, enable_checkpointing=True,
        enable_progress_bar=False)
    tr.fit(model, loader, val_dataloaders=val_loader)
    torch.save({"state_dict": model.value_module.state_dict(),
                "params": {k: str(v) for k, v in params.items()},
                "trial": trial_no, "max_steps": CONV_STEPS},
               f"{ckdir}/value_module.pt")
    st, cv = read_curve(f"{CONV_LOG_DIR}/{tag}/version_0/metrics.csv")
    cstep, plateau = detect_convergence(st, cv)
    flcb = lcb_of(cv)
    print(f"  {tag}: elapsed {(time.time()-t0)/60:.1f} min | "
          f"plateau≈{plateau:.3f} | converged@step={cstep} | "
          f"final-LCB={flcb:.3f}", flush=True)
    conv = {"tag": tag, "trial": trial_no, "method": params["method"],
            "params": {k: str(v) for k, v in params.items()},
            "plateau_reward": plateau, "convergence_step": cstep,
            "final_lcb": flcb, "ckpt_dir": ckdir,
            "steps": st.tolist(), "val_reward": cv.tolist()}
    json.dump(conv, open(f"{HERE}/conv_t{trial_no}.json", "w"), indent=2)
    del model, vm, tr, loader, ds
    gc.collect(); empty_cache()
    return conv


if CONV_TRIAL is not None:
    # Converge-only worker (spawned by the aggregator below).
    n = int(CONV_TRIAL)
    converge_one(n, confirm[n]["_params_raw"])
    raise SystemExit(0)

# Aggregator: best confirmed config per METHOD, converged in parallel.
per_method = {}
for tr_, v in confirm.items():
    m = v["_params_raw"]["method"]
    if v["lcb_mean"] > -50 and (
            m not in per_method
            or v["lcb_mean"] > confirm[per_method[m]]["lcb_mean"]):
        per_method[m] = tr_
to_converge = sorted(set(per_method.values()))
print(f"\nPHASE 3 — converge best-per-method in parallel ({CONV_STEPS} steps): "
      f"{[(m, f't{tr_}') for m, tr_ in per_method.items()]}\n", flush=True)

import subprocess  # noqa: E402

procs = []
for n in to_converge:
    pth = f"{HERE}/conv_t{n}.json"
    if os.path.exists(pth):
        os.remove(pth)
    env = dict(os.environ, OPT_CONV_TRIAL=str(n))
    procs.append((n, subprocess.Popen(
        ["python", f"{HERE}/optuna_fbrrt_bs4_sweep.py"], env=env)))
for n, pr in procs:
    rc = pr.wait()
    print(f"  converge t{n}: exit={rc}", flush=True)

convs = {}
for n in to_converge:
    pth = f"{HERE}/conv_t{n}.json"
    if os.path.exists(pth):
        convs[n] = json.load(open(pth))
if not convs:
    raise RuntimeError("no convergence runs produced results")
# Headline = converged run of the overall confirmed winner (fall back to the
# best converged plateau if the winner's method lost per-method selection).
conv = convs.get(best_trial) or max(convs.values(),
                                    key=lambda c: c["plateau_reward"])
tag = conv["tag"]


# ── Comparison vs the archived BS=4 winners (2026-05-28, pre-fix code) ──────
prior = {}
for pth, label in [
    (f"{ARCHIVE}/optuna_other_onpolicy_pipeline_results.json", None),
    (f"{ARCHIVE}/optuna_confirm_converge_results.json", None),
]:
    if os.path.exists(pth):
        d = json.load(open(pth))
        for k, vv in d.get("convergence", {}).items():
            prior[k] = vv
# off-policy converge results (stored under "comparison" -> "off_policy")
opp = f"{ARCHIVE}/optuna_offpolicy_pipeline_results.json"
if os.path.exists(opp):
    d = json.load(open(opp))
    op = d.get("comparison", {}).get("off_policy")
    if op and "plateau_reward" in op:
        prior["offpolicy_t0_converge"] = op
# amctl re-tune (flat convergence dict)
amc = f"{ARCHIVE}/optuna_amctl_bs4_sweep_results.json"
if os.path.exists(amc):
    d = json.load(open(amc))
    cv_ = d.get("convergence", {})
    if "plateau_reward" in cv_:
        prior["amctl_retuned_t" + str(cv_.get("trial", "?"))] = cv_

print("\n" + "=" * 88)
print("FBRRT (re-tuned, FIXED estimators) vs archived BS=4 winners (pre-fix code)")
print("=" * 88)
print(f"{'config':>48} | {'plateau':>9} | {'conv_step':>9} | {'final_LCB':>9}")
print("-" * 88)
for k, vv in prior.items():
    try:
        print(f"{k:>48} | {vv['plateau_reward']:>9.3f} | "
              f"{str(vv['convergence_step']):>9} | {vv['final_lcb']:>9.3f}")
    except (KeyError, TypeError):
        pass
for n, cc in sorted(convs.items()):
    print(f"{cc['tag']:>48} | {cc['plateau_reward']:>9.3f} | "
          f"{str(cc['convergence_step']):>9} | {cc['final_lcb']:>9.3f}")
print("=" * 88, flush=True)


# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
fig.suptitle("FBRRT family — Optuna re-tune @ BS=4 (fixed estimators, 2026-06-10)",
             fontsize=12, fontweight="bold")
ax = axes[0]
ax.set_title(f"Phase 2: confirmed configs LCB ({N_SEEDS} seeds)")
labs, ms, sds = [], [], []
for tr_, v in sorted(confirm.items(), key=lambda kv: kv[1]["lcb_mean"],
                     reverse=True):
    labs.append(f"t{tr_}\n{v['_params_raw']['method'].replace('fbrrt', 'f')}")
    ms.append(v["lcb_mean"]); sds.append(v["lcb_sd"])
xp = np.arange(len(labs))
ax.bar(xp, ms, yerr=sds, color="#16a085", capsize=4)
ax.set_xticks(xp); ax.set_xticklabels(labs, fontsize=8)
ax.set_ylabel("mean LCB"); ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
ax.set_title("Convergence vs archived BS=4 winners")
conv_palette = {"fbrrt": "#16a085", "fbrrt_td_lambda": "#e67e22",
                "fbrrt_cv": "#8e44ad"}
for n, cc in sorted(convs.items()):
    col = conv_palette.get(cc["method"], "#16a085")
    stc = np.array(cc["steps"]); cvc = np.array(cc["val_reward"])
    ax.plot(stc, cvc, color=col, alpha=0.25, lw=0.8)
    ax.plot(stc, pd.Series(cvc).rolling(8, min_periods=1).mean(), color=col,
            lw=2.0, label=f"{cc['method']} t{n} fixed "
                          f"(plateau={cc['plateau_reward']:.2f})")
palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b", "#9467bd"]
for (k, vv), col in zip(prior.items(), palette):
    try:
        ax.axhline(vv["plateau_reward"], color=col, ls="--", alpha=0.8,
                   label=f"{str(k).split('_t')[0][:26]} plateau={vv['plateau_reward']:.2f}")
    except (KeyError, TypeError):
        pass
ax.set_xlabel("training step"); ax.set_ylabel("val reward (mean)")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{HERE}/optuna_fbrrt2_bs4_sweep.png", dpi=140, bbox_inches="tight")
print(f"\nSaved: {HERE}/optuna_fbrrt2_bs4_sweep.png", flush=True)

json.dump(
    {"methods": METHODS,
     "n_trials": N_TRIALS,
     "chosen": [{"trial": t.number, "lcb": t.value,
                 "params": trial_params(t)} for t in chosen],
     "confirm": {k: {kk: vv for kk, vv in v.items() if kk != "_params_raw"}
                 for k, v in confirm.items()},
     "winner": {"trial": best_trial,
                **{kk: vv for kk, vv in best_conf.items()
                   if kk != "_params_raw"}},
     "convergence": {n: {k: v for k, v in cc.items()
                         if k not in ("steps", "val_reward")}
                     for n, cc in convs.items()},
     "convergence_curves": {n: {"steps": cc["steps"],
                                "val_reward": cc["val_reward"]}
                            for n, cc in convs.items()},
     "prior_comparison": {k: v for k, v in prior.items()
                          if isinstance(v, dict) and "plateau_reward" in v}},
    open(f"{HERE}/optuna_fbrrt2_bs4_sweep_results.json", "w"), indent=2,
    default=str)
print(f"Saved: {HERE}/optuna_fbrrt2_bs4_sweep_results.json")
print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min\nDone.")
