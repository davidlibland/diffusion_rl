#!/usr/bin/env python3
"""Combined Optuna pipeline for the OTHER on-policy methods.

`sampling_method` is itself an Optuna categorical over all 7 remaining
on-policy algorithms, with family-conditional hyperparameters:

Family A — log_tau / smc_value methods:
    one_step_bootstrap, ancestral_td_lambda, ancestral_mc_td_lambda
  design space: smc_type {kt_r,k_r,k_Vema,k_Vnograd,kV_plus_ltr}, k, l,
  ema_decay, lambda_eff (ancestral_* only — one_step has no λ)

Family B — FBRRT gradient-guided methods (no smc_value):
    fbrrt, fbrrt_td_lambda, fbrrt_cv, fbrrt_mc_z
  design space: alpha (guidance scale; 1=optimal on-policy, 0=base),
  entropy_lambda, branch, lambda_eff (fbrrt_td_lambda only)

Common: n_steps, off_policy_frac, mc_samples, lr, grad_decay (∇V weight decay).
Batch size fixed at 4.

Phase 1 sweep   : 100 trials, Hyperband, MAX_STEPS=5000.
Phase 2 confirm : top-3 overall (∪ best-per-family) × 5 seeds × 5000 steps.
Phase 3 converge: best Family-A config + best FBRRT config → 50000 steps,
                  serialized (best.ckpt/last.ckpt/value_module.pt).

Objective = detrended-SEM LCB over the last 20 val checkpoints (identical
to the on/off pipelines, for a direct comparison at BS=4).
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


# ── Shared setup ──────────────────────────────────────────────────────────
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


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def base_drift(x, t):
    return gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)


def reward_fn(x):
    return -10 * (x - c.to(x)).square().sum(dim=1)


def gmm_sample(n):
    k_ = np.random.choice(len(weights_np), size=n, p=weights_np)
    return means_np[k_] + sigmas_np[k_, np.newaxis] * np.random.randn(n, D)


def zero_log_tau(x, t):
    return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


class OnPolicyValueLive(OnPolicyValue):
    def drift(self, x, t, beta=1, use_ema=False):
        return super().drift(x, t, beta=beta, use_ema=use_ema)


FAMILY_A = ["one_step_bootstrap", "ancestral_td_lambda", "ancestral_mc_td_lambda"]
FAMILY_B = ["fbrrt", "fbrrt_td_lambda", "fbrrt_cv", "fbrrt_mc_z"]
ALL_METHODS = FAMILY_A + FAMILY_B
TD_A = {"ancestral_td_lambda", "ancestral_mc_td_lambda"}   # use lambda_eff
TD_B = {"fbrrt_td_lambda"}                                 # use lambda_eff


def make_smc_value(smc_type, k, l, model):
    if smc_type == "kt_r":
        return lambda x, t: k * t.reshape(-1) * reward_fn(x)
    if smc_type == "k_r":
        return lambda x, t: k * reward_fn(x)
    if smc_type == "k_Vema":
        return lambda x, t: k * model.ema(x, t).reshape(-1)
    if smc_type == "k_Vnograd":
        return lambda x, t: k * model.value_module(x, t).reshape(-1)
    if smc_type == "kV_plus_ltr":
        return lambda x, t: (k * model.value_module(x, t).reshape(-1)
                             + l * t.reshape(-1) * reward_fn(x))
    raise ValueError(smc_type)


# ── Constants ──────────────────────────────────────────────────────────────
LOG_DIR = "lightning_logs/optuna_other"
CONFIRM_DIR = "lightning_logs/optuna_other_confirm"
CONV_LOG_DIR = "lightning_logs/optuna_other_converge"
CKPT_DIR = "checkpoints/optuna_other_converge"
STUDY_DB = "sqlite:///notebooks/optuna_other.db"
STUDY_NAME = "other_onpolicy_lcb_v1"

BS = 4
DS_BATCH = 64
MAX_STEPS = int(os.environ.get("OPT_MAX_STEPS", 5000))
N_VAL = int(os.environ.get("OPT_N_VAL", 50))
LCB_TAIL = 20
LCB_Z = 1.645
N_TRIALS = int(os.environ.get("OPT_N_TRIALS", 100))
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


def sample_params(trial):
    """Family-conditional design space."""
    method = trial.suggest_categorical("method", ALL_METHODS)
    p = {
        "method": method,
        "n_steps": trial.suggest_int("n_steps", 10, 60),
        "off_policy_frac": trial.suggest_float("off_policy_frac", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
    }
    if trial.suggest_categorical("use_grad_decay", [True, False]):
        p["grad_decay"] = trial.suggest_float("grad_decay", 1e-5, 1e-1, log=True)
        p["use_grad_decay"] = True
    else:
        p["use_grad_decay"] = False

    if method in FAMILY_A:
        p["mc_samples"] = trial.suggest_int("mc_samples", 1, 24, log=True)
        p["smc_type"] = trial.suggest_categorical(
            "smc_type", ["kt_r", "k_r", "k_Vema", "k_Vnograd", "kV_plus_ltr"])
        p["k"] = trial.suggest_float("k", 1e-3, 1.0, log=True)
        if p["smc_type"] == "kV_plus_ltr":
            p["l"] = trial.suggest_float("l", 1e-3, 1.0, log=True)
        if p["smc_type"] == "k_Vema":
            p["ema_decay"] = trial.suggest_float("ema_decay", 0.90, 0.999)
        if method in TD_A:
            p["lambda_eff"] = trial.suggest_float("lambda_eff", 0.0, 1.0)
    else:  # Family B (FBRRT) — bounded particle budget
        p["mc_samples"] = trial.suggest_int("mc_samples_b", 1, 16, log=True)
        p["branch"] = trial.suggest_int("branch", 2, 10)
        p["alpha"] = trial.suggest_float("alpha", 0.0, 1.5)
        p["entropy_lambda"] = trial.suggest_float("entropy_lambda", 0.0, 2.0)
        if method in TD_B:
            p["lambda_eff"] = trial.suggest_float("lambda_eff_b", 0.0, 1.0)
    return p


def build(params, seed):
    L.seed_everything(seed, workers=True)
    method = params["method"]
    grad_decay = params["grad_decay"] if params.get("use_grad_decay") else None
    ema_decay = params.get("ema_decay", 0.99)
    vm = ValueNetwork(D, bias=bias_val)
    model = OnPolicyValueLive(
        base_score_module=base_drift, value_module=vm, reward_function=reward_fn,
        dim=D, a=a, lr=params["lr"], loss_type="quad",
        grad_decay=grad_decay, ema_decay=ema_decay,
    )
    # FBRRT computes grad_x v_theta on the first lazy data-gen, before
    # Lightning moves the module to the accelerator — move it now so the
    # value network and the (MPS) sample tensors share a device.
    model = model.to(DEVICE)
    if method in FAMILY_A:
        smc_value = make_smc_value(params["smc_type"], params["k"],
                                   params.get("l", 0.0), model)
    else:
        smc_value = zero_log_tau  # FBRRT ignores smc_value
    ds = OnPolicySMCDataset(
        dim=D, drift=base_drift, value=model.value_module,
        smc_value=smc_value, reward=reward_fn, device=DEVICE, a=a,
        batch_size=DS_BATCH, n_steps=params["n_steps"],
        mc_samples_per_step=params["mc_samples"],
        sampling_method=method,
        lambda_eff=params.get("lambda_eff", 0.1),
        branch=params.get("branch", 4),
        entropy_lambda=params.get("entropy_lambda", 1.0),
        fbrrt_alpha=params.get("alpha", 1.0),
        off_policy_frac=params["off_policy_frac"], include_t_zero=False,
        random_t=False, generating_function=gmm_sample,
    )
    loader = DataLoader(ds, batch_size=BS)
    return model, vm, ds, loader


def fmt(p):
    base = (f"{p['method']} mc={p['mc_samples']} ns={p['n_steps']} "
            f"ofp={p['off_policy_frac']:.2f} lr={p['lr']:.1e} "
            f"{'gd='+format(p['grad_decay'],'.1e') if p.get('use_grad_decay') else 'gd=off'}")
    if p["method"] in FAMILY_A:
        base += f" smc={p['smc_type']} k={p['k']:.3f}"
        if "lambda_eff" in p:
            base += f" lam={p['lambda_eff']:.2f}"
    else:
        base += (f" br={p['branch']} alpha={p['alpha']:.2f} "
                 f"ent={p['entropy_lambda']:.2f}")
        if "lambda_eff" in p:
            base += f" lam={p['lambda_eff']:.2f}"
    return base


# ── Phase 1: sweep ────────────────────────────────────────────────────────
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
        print(f"  {name}: pruned [{p['method']}] "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        raise
    except (RuntimeError, ValueError) as e:
        del model, vm, trainer, loader, ds
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"  {name}: error [{p['method']}] {type(e).__name__} -> -100",
              flush=True)
        return -100.0
    st, cv = read_curve(csv)
    del model, vm, trainer, loader, ds
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    lcb = lcb_of(cv)
    print(f"  {name}: LCB={lcb:.3f}  [{fmt(p)}]  "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    return lcb


t_start = time.time()
sampler = TPESampler(multivariate=True, group=True, seed=42)
pruner = HyperbandPruner(min_resource=500, max_resource=MAX_STEPS,
                         reduction_factor=3)
study = optuna.create_study(
    study_name=STUDY_NAME, storage=STUDY_DB, load_if_exists=True,
    direction="maximize", sampler=sampler, pruner=pruner)
n_done = len([t for t in study.trials if t.state.is_finished()])
print(f"PHASE 1 — combined sweep (7 methods)  device={DEVICE}  "
      f"done={n_done} remaining={max(0, N_TRIALS-n_done)}", flush=True)


def _cb(study, trial):
    done = len([t for t in study.trials if t.state.is_finished()])
    try:
        best = study.best_value
        bm = study.best_trial.params.get("method", "?")
    except ValueError:
        best, bm = float("nan"), "?"
    print(f"[{done}/{N_TRIALS}] elapsed={(time.time()-t_start)/60:.1f} min "
          f"| best LCB={best:.3f} ({bm})", flush=True)


study.optimize(objective, n_trials=max(0, N_TRIALS - n_done),
               callbacks=[_cb], gc_after_trial=True)


def trial_params(t):
    """Reconstruct the full params dict from a finished trial."""
    pr = dict(t.params)
    m = pr["method"]
    out = {"method": m, "n_steps": pr["n_steps"],
           "off_policy_frac": pr["off_policy_frac"], "lr": pr["lr"],
           "use_grad_decay": pr.get("use_grad_decay", False)}
    if out["use_grad_decay"]:
        out["grad_decay"] = pr["grad_decay"]
    if m in FAMILY_A:
        out["mc_samples"] = pr["mc_samples"]
        out["smc_type"] = pr["smc_type"]; out["k"] = pr["k"]
        if pr["smc_type"] == "kV_plus_ltr":
            out["l"] = pr["l"]
        if pr["smc_type"] == "k_Vema":
            out["ema_decay"] = pr["ema_decay"]
        if m in TD_A:
            out["lambda_eff"] = pr["lambda_eff"]
    else:
        out["mc_samples"] = pr["mc_samples_b"]
        out["branch"] = pr["branch"]; out["alpha"] = pr["alpha"]
        out["entropy_lambda"] = pr["entropy_lambda"]
        if m in TD_B:
            out["lambda_eff"] = pr["lambda_eff_b"]
    return out


comp = [t for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None]
comp.sort(key=lambda t: t.value, reverse=True)

# top-3 overall ∪ best Family-A ∪ best Family-B (so each family is converged)
chosen = list(comp[:3])
for fam in (FAMILY_A, FAMILY_B):
    famc = [t for t in comp if t.params.get("method") in fam]
    if famc and famc[0] not in chosen:
        chosen.append(famc[0])
seen = set()
chosen = [t for t in chosen if not (t.number in seen or seen.add(t.number))]

print("\n" + "=" * 80)
print("Selected for 5-seed confirm (top-3 ∪ best/family):")
for t in chosen:
    print(f"  trial {t.number:>3}  LCB={t.value:>8.3f}  [{fmt(trial_params(t))}]")
print("=" * 80, flush=True)
json.dump([{"trial": t.number, "lcb": t.value, "params": trial_params(t)}
           for t in chosen],
          open("notebooks/optuna_other_top.json", "w"), indent=2)


# ── Phase 2: confirm ──────────────────────────────────────────────────────
print("\nPHASE 2 — confirm × 5 seeds × 5000 steps\n", flush=True)
confirm = {}
for t in chosen:
    params = trial_params(t)
    trial = t.number
    fam = "A" if params["method"] in FAMILY_A else "B"
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
                max_steps=MAX_STEPS,
                val_check_interval=max(1, MAX_STEPS // N_VAL),
                logger=logger, enable_checkpointing=False,
                enable_progress_bar=False)
            tr.fit(model, loader, val_dataloaders=val_loader)
            del model, vm, tr, loader, ds
        except (RuntimeError, ValueError) as e:
            print(f"  {name}: error {type(e).__name__}", flush=True)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        st, cv = read_curve(csv)
        lcb = lcb_of(cv) if len(cv) else -100.0
        best = float(cv.max()) if len(cv) else -100.0
        seed_lcbs.append(lcb); seed_bests.append(best)
        print(f"  {name}: LCB={lcb:.3f} best={best:.3f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    seed_lcbs = np.array(seed_lcbs); seed_bests = np.array(seed_bests)
    confirm[trial] = {
        "params": params, "family": fam,
        "lcb_mean": float(seed_lcbs.mean()),
        "lcb_sd": float(seed_lcbs.std(ddof=1)),
        "lcb_values": seed_lcbs.tolist(),
        "best_mean": float(seed_bests.mean()),
        "best_sd": float(seed_bests.std(ddof=1)),
    }
    print(f"  >>> t{trial} [{params['method']}]: LCB "
          f"{seed_lcbs.mean():.3f} ± {seed_lcbs.std(ddof=1):.3f}  "
          f"best {seed_bests.mean():.3f} ± {seed_bests.std(ddof=1):.3f}",
          flush=True)

json.dump(confirm, open("notebooks/optuna_other_confirm_results.json", "w"),
          indent=2)

winners = {}
for fam in ("A", "B"):
    cand = {tr: v for tr, v in confirm.items() if v["family"] == fam}
    if cand:
        bt = max(cand, key=lambda tr: cand[tr]["lcb_mean"])
        winners[fam] = (bt, cand[bt])
print("\n" + "=" * 80)
for fam, (bt, v) in winners.items():
    print(f"Family {fam} winner: trial {bt} [{v['params']['method']}]  "
          f"LCB {v['lcb_mean']:.3f} ± {v['lcb_sd']:.3f}")
print("=" * 80, flush=True)


# ── Phase 3: converge winners (best Family-A + best FBRRT) ─────────────────
print("\nPHASE 3 — converge family winners (50000 steps, serialized)\n",
      flush=True)


def detect_convergence(steps, curve, win=8):
    if len(curve) < win + 4:
        return None, float(curve[-1]) if len(curve) else float("nan")
    sm = pd.Series(curve).rolling(win, min_periods=1).mean().to_numpy()
    tail = sm[-max(4, len(sm) // 5):]
    plateau = float(tail.mean())
    noise = float(np.std(curve[len(curve) // 2:] - sm[len(sm) // 2:]))
    thresh = plateau - 0.5 * noise
    conv_step = None
    for i in range(len(sm)):
        if sm[i] >= thresh and np.all(sm[i:] >= plateau - noise):
            conv_step = int(steps[i]); break
    return conv_step, plateau


conv_summary = {}
for fam, (bt, v) in winners.items():
    params = v["params"]
    tag = f"fam{fam}_t{bt}_{params['method']}_converge"
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
        callbacks=[ckpt], logger=logger,
        enable_checkpointing=True, enable_progress_bar=False)
    tr.fit(model, loader, val_dataloaders=val_loader)
    torch.save({"state_dict": model.value_module.state_dict(),
                "params": params, "trial": bt, "family": fam,
                "max_steps": CONV_STEPS}, f"{ckdir}/value_module.pt")
    st, cv = read_curve(f"{CONV_LOG_DIR}/{tag}/version_0/metrics.csv")
    cstep, plateau = detect_convergence(st, cv)
    flcb = lcb_of(cv)
    print(f"  elapsed {(time.time()-t0)/60:.1f} min | plateau≈{plateau:.3f} "
          f"| converged@step={cstep} | final-LCB={flcb:.3f}", flush=True)
    print(f"  ckpts: {ckdir}/{{best.ckpt,last.ckpt,value_module.pt}}",
          flush=True)
    conv_summary[tag] = {
        "family": fam, "trial": bt, "params": params,
        "plateau_reward": plateau, "convergence_step": cstep,
        "final_lcb": flcb, "ckpt_dir": ckdir,
        "steps": st.tolist(), "val_reward": cv.tolist()}
    del model, vm, tr, loader, ds
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── Comparison vs prior on/off convergence ────────────────────────────────
prior = {}
for pth, lbl in [("notebooks/optuna_confirm_converge_results.json", "on"),
                 ("notebooks/optuna_offpolicy_pipeline_results.json", "off")]:
    if os.path.exists(pth):
        d = json.load(open(pth))
        if lbl == "on":
            for k, vv in d.get("convergence", {}).items():
                prior[k] = vv
        else:
            cp = d.get("comparison", {}).get("off_policy")
            if cp:
                prior["offpolicy_converge"] = cp

print("\n" + "=" * 84)
print("ALL METHODS @ BS=4 (converged)")
print("=" * 84)
print(f"{'config':>40} | {'plateau':>9} | {'conv_step':>9} | {'final_LCB':>9}")
print("-" * 84)
for k, vv in prior.items():
    print(f"{k:>40} | {vv['plateau_reward']:>9.3f} | "
          f"{str(vv['convergence_step']):>9} | {vv['final_lcb']:>9.3f}")
for tag, d in conv_summary.items():
    print(f"{tag:>40} | {d['plateau_reward']:>9.3f} | "
          f"{str(d['convergence_step']):>9} | {d['final_lcb']:>9.3f}")
print("=" * 84, flush=True)


# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
fig.suptitle("Other on-policy methods — Optuna pipeline @ BS=4",
             fontsize=12, fontweight="bold")
ax = axes[0]
ax.set_title("Phase 2: confirmed configs LCB (5 seeds)")
labs, ms, sds, cols = [], [], [], []
for tr, v in sorted(confirm.items(), key=lambda kv: kv[1]["lcb_mean"],
                     reverse=True):
    labs.append(f"t{tr}\n{v['params']['method'][:10]}")
    ms.append(v["lcb_mean"]); sds.append(v["lcb_sd"])
    cols.append("#9467bd" if v["family"] == "A" else "#8c564b")
xp = np.arange(len(labs))
ax.bar(xp, ms, yerr=sds, color=cols, capsize=4)
ax.set_xticks(xp); ax.set_xticklabels(labs, fontsize=7)
ax.set_ylabel("mean LCB"); ax.grid(True, alpha=0.3, axis="y")
ax.plot([], [], color="#9467bd", lw=4, label="Family A (smc_value)")
ax.plot([], [], color="#8c564b", lw=4, label="Family B (FBRRT)")
ax.legend(fontsize=9)

ax = axes[1]
ax.set_title("Convergence: family winners vs prior on/off")
for tag, d in conv_summary.items():
    col = "#9467bd" if d["family"] == "A" else "#8c564b"
    st = np.array(d["steps"]); cv = np.array(d["val_reward"])
    ax.plot(st, cv, color=col, alpha=0.30, lw=1.0)
    sm = pd.Series(cv).rolling(8, min_periods=1).mean()
    ax.plot(st, sm, color=col, lw=2.0,
            label=f"{d['params']['method']} (conv@{d['convergence_step']})")
for k, vv in prior.items():
    if "td_lambda" in k:
        col = "#1f77b4"
    elif "mc" in k:
        col = "#ff7f0e"
    else:
        col = "#2ca02c"
    ax.axhline(vv["plateau_reward"], color=col, ls="--", alpha=0.8,
               label=f"{k.split('_t')[0][:18]} plateau={vv['plateau_reward']:.2f}")
ax.set_xlabel("training step"); ax.set_ylabel("val reward (mean)")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("notebooks/optuna_other_onpolicy_pipeline.png", dpi=140,
            bbox_inches="tight")
print("\nSaved: notebooks/optuna_other_onpolicy_pipeline.png", flush=True)

json.dump(
    {"chosen": [{"trial": t.number, "lcb": t.value,
                 "params": trial_params(t)} for t in chosen],
     "confirm": confirm,
     "winners": {fam: {"trial": bt, **v} for fam, (bt, v) in winners.items()},
     "convergence": {k: {kk: vv for kk, vv in d.items()
                         if kk not in ("steps", "val_reward")}
                     for k, d in conv_summary.items()},
     "prior_comparison": prior},
    open("notebooks/optuna_other_onpolicy_pipeline_results.json", "w"),
    indent=2)
print("Saved: notebooks/optuna_other_onpolicy_pipeline_results.json")
print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min")
print("Done.")
