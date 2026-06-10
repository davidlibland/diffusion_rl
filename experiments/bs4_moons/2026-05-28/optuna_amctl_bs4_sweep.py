#!/usr/bin/env python3
"""Optuna sweep for `ancestral_mc_td_lambda` on the moons BS=4 benchmark.

This is a focused, single-method companion to
``optuna_other_onpolicy_pipeline.py`` (which lumped ``ancestral_mc_td_lambda``
in with the rest of "Family A").  It exists because the multi-step / duplicate
averaging in ``ancestral_mc_td_lambda`` was just corrected (the twist used to
leak into the targets); a dedicated sweep re-tunes the method under the fixed
estimator and explores a *more general* ``log_tau`` (a.k.a. ``smc_value``)
design space.

log_tau / smc_value design space
--------------------------------
``log_tau`` is the twist used only for SMC resampling; it cancels out of the
(now-unbiased) targets, so it is purely a variance-reduction / proposal knob.
Following the BS=4 work we allow it to be ANY non-negative linear combination
of four basis functions:

    log_tau(x,t) = cr * r(x)              # reward
                 + ctr * t * r(x)         # time-annealed reward
                 + cV * V_theta(x,t)      # live value network (no grad)
                 + cVema * V_ema(x,t)     # EMA value network

Each term is independently toggled on/off (so the classic single-term twists
``k*r``, ``k*t*r``, ``k*V``, ``k*V_ema`` are all in-distribution), and active
coefficients are sampled log-uniformly in [1e-3, 3].  Coefficients are
non-negative on purpose: r and V are both larger for better states, so a
positive weight is what concentrates SMC mass where it matters.  ``ema_decay``
is sampled only when the EMA term is active.

Other swept hyperparameters
---------------------------
lambda_eff (TD(λ) mixing), mc_samples (particles per seed), n_steps,
off_policy_frac (off-policy anchoring), lr, grad_decay (∇V weight decay,
toggled), loss_type ∈ {quad, mse}.  Gradient batch size fixed at BS=4 and
``include_t_zero`` is irrelevant here (ancestral_mc always emits the t=0/t=1
generations internally).

Objective
---------
Detrended-SEM lower confidence bound (LCB) over the last ``LCB_TAIL`` of
``N_VAL`` validation checkpoints — the same windowed lower-bound metric the
rest of the BS=4 study optimises (more reliable than raw tail reward), so the
numbers here are directly comparable to the summary table.

Pipeline (env-var tunable budgets; defaults match the BS=4 study)
----------------------------------------------------------------
Phase 1 sweep   : OPT_N_TRIALS trials, TPE + Hyperband, OPT_MAX_STEPS steps.
Phase 2 confirm : top-OPT_TOPK trials × OPT_N_SEEDS seeds × OPT_MAX_STEPS steps.
Phase 3 converge: best confirmed config → OPT_CONV_STEPS steps, serialized
                  (best.ckpt / last.ckpt / value_module.pt).

For a fast smoke test:
    OPT_N_TRIALS=2 OPT_MAX_STEPS=200 OPT_N_VAL=4 OPT_TOPK=1 OPT_N_SEEDS=2 \
    OPT_CONV_STEPS=300 python experiments/bs4_moons/optuna_amctl_bs4_sweep.py
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
from optuna.trial import TrialState

from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork


# ── Device (CUDA on this box; falls back to MPS/CPU) ───────────────────────
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


# ── log_tau (smc_value): general non-negative linear combination ───────────
SMC_BASES = ("r", "tr", "V", "Vema")  # reward, t*reward, value, ema-value


def make_smc_value(spec, model):
    """spec maps a subset of {cr, ctr, cV, cVema} -> coefficient (>0)."""
    cr = spec.get("cr"); ctr = spec.get("ctr")
    cV = spec.get("cV"); cVema = spec.get("cVema")

    def smc(x, t):
        out = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        if cr is not None or ctr is not None:
            r = reward_fn(x)
            if cr is not None:
                out = out + cr * r
            if ctr is not None:
                out = out + ctr * t.reshape(-1) * r
        if cV is not None:
            out = out + cV * model.value_module(x, t).reshape(-1)
        if cVema is not None:
            out = out + cVema * model.ema(x, t).reshape(-1)
        return out

    return smc


# ── Constants ──────────────────────────────────────────────────────────────
LOG_DIR = "lightning_logs/optuna_amctl"
CONFIRM_DIR = "lightning_logs/optuna_amctl_confirm"
CONV_LOG_DIR = "lightning_logs/optuna_amctl_converge"
CKPT_DIR = "checkpoints/optuna_amctl_converge"
STUDY_DB = "sqlite:///experiments/bs4_moons/optuna_amctl.db"
STUDY_NAME = "amctl_bs4_lcb_v1"
METHOD = "ancestral_mc_td_lambda"

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
        "n_steps": trial.suggest_int("n_steps", 10, 60),
        "mc_samples": trial.suggest_int("mc_samples", 2, 32, log=True),
        "off_policy_frac": trial.suggest_float("off_policy_frac", 0.0, 0.5),
        "lambda_eff": trial.suggest_float("lambda_eff", 0.0, 1.0),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "loss_type": trial.suggest_categorical("loss_type", ["quad", "mse"]),
    }
    if trial.suggest_categorical("use_grad_decay", [True, False]):
        p["grad_decay"] = trial.suggest_float("grad_decay", 1e-5, 1e-1, log=True)
        p["use_grad_decay"] = True
    else:
        p["use_grad_decay"] = False

    # log_tau = non-negative linear combination of {r, t*r, V, V_ema}.
    use = {b: trial.suggest_categorical(f"use_{b}", [True, False]) for b in SMC_BASES}
    if not any(use.values()):           # never let the twist vanish
        use["r"] = True
    spec = {}
    if use["r"]:
        spec["cr"] = trial.suggest_float("cr", 1e-3, 3.0, log=True)
    if use["tr"]:
        spec["ctr"] = trial.suggest_float("ctr", 1e-3, 3.0, log=True)
    if use["V"]:
        spec["cV"] = trial.suggest_float("cV", 1e-3, 3.0, log=True)
    if use["Vema"]:
        spec["cVema"] = trial.suggest_float("cVema", 1e-3, 3.0, log=True)
    p["smc_spec"] = spec
    if use["Vema"]:
        p["ema_decay"] = trial.suggest_float("ema_decay", 0.90, 0.999)
    return p


def trial_params(t):
    """Reconstruct the full params dict from a finished trial (mirror of sample)."""
    pr = dict(t.params)
    out = {
        "n_steps": pr["n_steps"], "mc_samples": pr["mc_samples"],
        "off_policy_frac": pr["off_policy_frac"], "lambda_eff": pr["lambda_eff"],
        "lr": pr["lr"], "loss_type": pr["loss_type"],
        "use_grad_decay": pr.get("use_grad_decay", False),
    }
    if out["use_grad_decay"]:
        out["grad_decay"] = pr["grad_decay"]
    use = {b: pr.get(f"use_{b}", False) for b in SMC_BASES}
    if not any(use.values()):
        use["r"] = True
    spec = {}
    if use["r"]:
        spec["cr"] = pr["cr"]
    if use["tr"]:
        spec["ctr"] = pr["ctr"]
    if use["V"]:
        spec["cV"] = pr["cV"]
    if use["Vema"]:
        spec["cVema"] = pr["cVema"]
        out["ema_decay"] = pr["ema_decay"]
    out["smc_spec"] = spec
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
    smc_value = make_smc_value(params["smc_spec"], model)
    ds = OnPolicySMCDataset(
        dim=D, drift=base_drift, value=model.value_module,
        smc_value=smc_value, reward=reward_fn, device=DEVICE, a=a,
        batch_size=DS_BATCH, n_steps=params["n_steps"],
        mc_samples_per_step=params["mc_samples"],
        sampling_method=METHOD, lambda_eff=params["lambda_eff"],
        off_policy_frac=params["off_policy_frac"], include_t_zero=False,
        random_t=False, generating_function=gmm_sample,
    )
    loader = DataLoader(ds, batch_size=BS)
    return model, vm, ds, loader


def fmt_spec(spec):
    parts = []
    for key, lbl in (("cr", "r"), ("ctr", "t·r"), ("cV", "V"), ("cVema", "Vema")):
        if key in spec:
            parts.append(f"{spec[key]:.3f}·{lbl}")
    return "+".join(parts) if parts else "(none)"


def fmt(p):
    s = (f"{METHOD} mc={p['mc_samples']} ns={p['n_steps']} "
         f"lam={p['lambda_eff']:.2f} ofp={p['off_policy_frac']:.2f} "
         f"lr={p['lr']:.1e} loss={p['loss_type']} "
         f"{'gd='+format(p['grad_decay'],'.1e') if p.get('use_grad_decay') else 'gd=off'} "
         f"tau=[{fmt_spec(p['smc_spec'])}]")
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
        print(f"  {name}: error {type(e).__name__} -> -100", flush=True)
        return -100.0
    st, cv = read_curve(csv)
    del model, vm, trainer, loader, ds
    gc.collect(); empty_cache()
    lcb = lcb_of(cv)
    print(f"  {name}: LCB={lcb:.3f}  [{fmt(p)}]  {(time.time()-t0)/60:.1f} min",
          flush=True)
    return lcb


t_start = time.time()
sampler = TPESampler(multivariate=True, group=True, seed=42)
pruner = HyperbandPruner(min_resource=500, max_resource=MAX_STEPS,
                         reduction_factor=3)
study = optuna.create_study(
    study_name=STUDY_NAME, storage=STUDY_DB, load_if_exists=True,
    direction="maximize", sampler=sampler, pruner=pruner)
n_done = len([t for t in study.trials if t.state.is_finished()])
print(f"PHASE 1 — {METHOD} sweep  device={DEVICE}  "
      f"done={n_done} remaining={max(0, N_TRIALS-n_done)}", flush=True)


def _cb(study, trial):
    done = len([t for t in study.trials if t.state.is_finished()])
    try:
        best = study.best_value
    except ValueError:
        best = float("nan")
    print(f"[{done}/{N_TRIALS}] elapsed={(time.time()-t_start)/60:.1f} min "
          f"| best LCB={best:.3f}", flush=True)


study.optimize(objective, n_trials=max(0, N_TRIALS - n_done),
               callbacks=[_cb], gc_after_trial=True)

comp = [t for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None]
comp.sort(key=lambda t: t.value, reverse=True)
chosen = comp[:TOPK]

print("\n" + "=" * 80)
print(f"Selected for {N_SEEDS}-seed confirm (top-{TOPK}):")
for t in chosen:
    print(f"  trial {t.number:>3}  LCB={t.value:>8.3f}  [{fmt(trial_params(t))}]")
print("=" * 80, flush=True)
json.dump([{"trial": t.number, "lcb": t.value, "params": trial_params(t)}
           for t in chosen],
          open("experiments/bs4_moons/optuna_amctl_top.json", "w"), indent=2)


# ── Phase 2: confirm ───────────────────────────────────────────────────────
print(f"\nPHASE 2 — confirm × {N_SEEDS} seeds × {MAX_STEPS} steps\n", flush=True)
confirm = {}
for t in chosen:
    params = trial_params(t); trial = t.number
    seed_lcbs, seed_bests = [], []
    for s in range(N_SEEDS):
        name = f"t{trial}_seed{s:02d}"
        csv = f"{CONFIRM_DIR}/{name}/version_0/metrics.csv"
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
            print(f"  {name}: error {type(e).__name__}", flush=True)
        gc.collect(); empty_cache()
        st, cv = read_curve(csv)
        lcb = lcb_of(cv) if len(cv) else -100.0
        best = float(cv.max()) if len(cv) else -100.0
        seed_lcbs.append(lcb); seed_bests.append(best)
        print(f"  {name}: LCB={lcb:.3f} best={best:.3f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    seed_lcbs = np.array(seed_lcbs); seed_bests = np.array(seed_bests)
    confirm[trial] = {
        "params": params,
        "lcb_mean": float(seed_lcbs.mean()),
        "lcb_sd": float(seed_lcbs.std(ddof=1)) if len(seed_lcbs) > 1 else 0.0,
        "lcb_values": seed_lcbs.tolist(),
        "best_mean": float(seed_bests.mean()),
        "best_sd": float(seed_bests.std(ddof=1)) if len(seed_bests) > 1 else 0.0,
    }
    print(f"  >>> t{trial}: LCB {seed_lcbs.mean():.3f} ± "
          f"{confirm[trial]['lcb_sd']:.3f}  best {seed_bests.mean():.3f} ± "
          f"{confirm[trial]['best_sd']:.3f}", flush=True)

json.dump(confirm, open("experiments/bs4_moons/optuna_amctl_confirm_results.json", "w"),
          indent=2)

best_trial = max(confirm, key=lambda tr: confirm[tr]["lcb_mean"])
best_conf = confirm[best_trial]
print("\n" + "=" * 80)
print(f"Confirmed winner: trial {best_trial}  "
      f"LCB {best_conf['lcb_mean']:.3f} ± {best_conf['lcb_sd']:.3f}")
print(f"  {fmt(best_conf['params'])}")
print("=" * 80, flush=True)


# ── Phase 3: converge winner ───────────────────────────────────────────────
print(f"\nPHASE 3 — converge winner ({CONV_STEPS} steps, serialized)\n",
      flush=True)


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


params = best_conf["params"]
tag = f"amctl_t{best_trial}_converge"
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
                       monitor="val_reward_mean", mode="max", filename="best")
tr = L.Trainer(
    max_steps=CONV_STEPS, val_check_interval=CONV_VAL_EVERY,
    callbacks=[ckpt], logger=logger, enable_checkpointing=True,
    enable_progress_bar=False)
tr.fit(model, loader, val_dataloaders=val_loader)
torch.save({"state_dict": model.value_module.state_dict(), "params": params,
            "trial": best_trial, "max_steps": CONV_STEPS},
           f"{ckdir}/value_module.pt")
st, cv = read_curve(f"{CONV_LOG_DIR}/{tag}/version_0/metrics.csv")
cstep, plateau = detect_convergence(st, cv)
flcb = lcb_of(cv)
print(f"  elapsed {(time.time()-t0)/60:.1f} min | plateau≈{plateau:.3f} "
      f"| converged@step={cstep} | final-LCB={flcb:.3f}", flush=True)
print(f"  ckpts: {ckdir}/{{best.ckpt,last.ckpt,value_module.pt}}", flush=True)
conv = {"trial": best_trial, "params": params, "plateau_reward": plateau,
        "convergence_step": cstep, "final_lcb": flcb, "ckpt_dir": ckdir,
        "steps": st.tolist(), "val_reward": cv.tolist()}
del model, vm, tr, loader, ds
gc.collect(); empty_cache()


# ── Comparison vs the prior BS=4 winners ───────────────────────────────────
prior = {}
op = "experiments/bs4_moons/optuna_other_onpolicy_pipeline_results.json"
if os.path.exists(op):
    d = json.load(open(op))
    for k, vv in d.get("convergence", {}).items():
        prior[k] = vv
onp = "experiments/bs4_moons/optuna_confirm_converge_results.json"
if os.path.exists(onp):
    d = json.load(open(onp))
    for k, vv in d.get("convergence", {}).items():
        prior[k] = vv

print("\n" + "=" * 84)
print("ancestral_mc_td_lambda (re-tuned, fixed estimator) vs prior BS=4 winners")
print("=" * 84)
print(f"{'config':>44} | {'plateau':>9} | {'conv_step':>9} | {'final_LCB':>9}")
print("-" * 84)
for k, vv in prior.items():
    print(f"{k:>44} | {vv['plateau_reward']:>9.3f} | "
          f"{str(vv['convergence_step']):>9} | {vv['final_lcb']:>9.3f}")
print(f"{tag:>44} | {conv['plateau_reward']:>9.3f} | "
      f"{str(conv['convergence_step']):>9} | {conv['final_lcb']:>9.3f}")
print("=" * 84, flush=True)


# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
fig.suptitle("ancestral_mc_td_lambda — Optuna sweep @ BS=4 (fixed estimator)",
             fontsize=12, fontweight="bold")
ax = axes[0]
ax.set_title(f"Phase 2: confirmed configs LCB ({N_SEEDS} seeds)")
labs, ms, sds = [], [], []
for tr_, v in sorted(confirm.items(), key=lambda kv: kv[1]["lcb_mean"],
                     reverse=True):
    labs.append(f"t{tr_}"); ms.append(v["lcb_mean"]); sds.append(v["lcb_sd"])
xp = np.arange(len(labs))
ax.bar(xp, ms, yerr=sds, color="#9467bd", capsize=4)
ax.set_xticks(xp); ax.set_xticklabels(labs, fontsize=8)
ax.set_ylabel("mean LCB"); ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
ax.set_title("Convergence vs prior BS=4 winners")
stc = np.array(conv["steps"]); cvc = np.array(conv["val_reward"])
ax.plot(stc, cvc, color="#9467bd", alpha=0.30, lw=1.0)
ax.plot(stc, pd.Series(cvc).rolling(8, min_periods=1).mean(), color="#9467bd",
        lw=2.0, label=f"amctl re-tuned (conv@{conv['convergence_step']}, "
                      f"plateau={conv['plateau_reward']:.2f})")
palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
for (k, vv), col in zip(prior.items(), palette):
    ax.axhline(vv["plateau_reward"], color=col, ls="--", alpha=0.8,
               label=f"{k.split('_t')[0][:22]} plateau={vv['plateau_reward']:.2f}")
ax.set_xlabel("training step"); ax.set_ylabel("val reward (mean)")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("experiments/bs4_moons/optuna_amctl_bs4_sweep.png", dpi=140, bbox_inches="tight")
print("\nSaved: experiments/bs4_moons/optuna_amctl_bs4_sweep.png", flush=True)

json.dump(
    {"method": METHOD,
     "chosen": [{"trial": t.number, "lcb": t.value,
                 "params": trial_params(t)} for t in chosen],
     "confirm": confirm,
     "winner": {"trial": best_trial, **best_conf},
     "convergence": {k: v for k, v in conv.items()
                     if k not in ("steps", "val_reward")},
     "convergence_curve": {"steps": conv["steps"],
                           "val_reward": conv["val_reward"]},
     "prior_comparison": prior},
    open("experiments/bs4_moons/optuna_amctl_bs4_sweep_results.json", "w"), indent=2)
print("Saved: experiments/bs4_moons/optuna_amctl_bs4_sweep_results.json")
print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min\nDone.")
