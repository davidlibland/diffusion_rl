#!/usr/bin/env python3
"""Optuna sweep for on-policy SSMC / SSMC-TD(λ), batch size fixed at 4.

Objective (maximize): lower confidence bound on the *converged mean*
validation reward.  Over the last ~20 validation checkpoints we OLS-detrend
the tail (so a still-improving run isn't penalised for its slope), estimate
the residual noise σ, and report
    LCB = tail_mean − 1.645 · σ_detrended / √n_tail
(single seed; the val cadence supplies the tail samples).  6 checkpoints
was shown to be too noisy — see notebooks/ check on past runs.

Pruning: Hyperband on the val_reward curve (resource = training step).

Search space
  method            {single_seed_mc, single_seed_td_lambda}
  n_steps           int 10–60
  random_t          {True, False}        uniform vs random sorted t-grid
  off_policy_frac   float 0–0.5
  smc_type          {kt_r, k_r, k_Vema, k_Vnograd, kV_plus_ltr}
  k                 loguniform 1e-3–1
  l                 loguniform 1e-3–1    (only kV_plus_ltr)
  ema_decay         0.90–0.999           (only k_Vema; tuned for step count)
  mc_samples        int 1–24 (log)
  lambda_eff        float 0–1            (only td_lambda)
  lr                loguniform 1e-4–3e-3
  grad_decay        {off} ∪ loguniform 1e-5–1e-1   (weight decay on ∇_x V)

Drift/inference always use the live value (OnPolicyValueLive, no EMA).
include_t_zero=False throughout (degenerate (x=0,t=0) point — see prior work).
"""

import gc
import json
import os
import time
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
from einops import reduce
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

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


class OnPolicyValueLive(OnPolicyValue):
    def drift(self, x, t, beta=1, use_ema=False):
        return super().drift(x, t, beta=beta, use_ema=use_ema)


# ── Constants ──────────────────────────────────────────────────────────────
LOG_DIR = "lightning_logs/optuna_onpolicy"
STUDY_DB = "sqlite:///notebooks/optuna_onpolicy.db"
STUDY_NAME = "onpolicy_lcb_v1"

BS = 4                       # gradient batch size (the constraint)
DS_BATCH = 64                # internal SMC trajectory batch (regen efficiency)
MAX_STEPS = int(os.environ.get("OPT_MAX_STEPS", 5000))
N_VAL = int(os.environ.get("OPT_N_VAL", 50))   # validation checkpoints / run
LCB_TAIL = 20                # # of trailing val points for the LCB
LCB_Z = 1.645
N_TRIALS = int(os.environ.get("OPT_N_TRIALS", 100))

all_rewards = reward_fn(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()
val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


# ── Optuna pruning callback ───────────────────────────────────────────────
class OptunaPruning(Callback):
    def __init__(self, trial, monitor="val_reward_mean"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        m = trainer.callback_metrics.get(self.monitor)
        if m is None:
            return
        step = int(trainer.global_step)
        self.trial.report(float(m), step)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"pruned at step {step}")


def _read_val_curve(csv_path):
    import pandas as pd
    if not os.path.exists(csv_path):
        return np.array([])
    df = pd.read_csv(csv_path)
    if "val_reward_mean" not in df.columns:
        return np.array([])
    return df.dropna(subset=["val_reward_mean"])["val_reward_mean"].to_numpy()


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


def objective(trial):
    method = trial.suggest_categorical(
        "method", ["single_seed_mc", "single_seed_td_lambda"])
    n_steps = trial.suggest_int("n_steps", 10, 60)
    random_t = trial.suggest_categorical("random_t", [True, False])
    off_policy_frac = trial.suggest_float("off_policy_frac", 0.0, 0.5)
    smc_type = trial.suggest_categorical(
        "smc_type", ["kt_r", "k_r", "k_Vema", "k_Vnograd", "kV_plus_ltr"])
    k = trial.suggest_float("k", 1e-3, 1.0, log=True)
    l = trial.suggest_float("l", 1e-3, 1.0, log=True) \
        if smc_type == "kV_plus_ltr" else 0.0
    ema_decay = trial.suggest_float("ema_decay", 0.90, 0.999) \
        if smc_type == "k_Vema" else 0.99
    mc_samples = trial.suggest_int("mc_samples", 1, 24, log=True)
    lambda_eff = trial.suggest_float("lambda_eff", 0.0, 1.0) \
        if method == "single_seed_td_lambda" else 0.1
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    if trial.suggest_categorical("use_grad_decay", [True, False]):
        grad_decay = trial.suggest_float("grad_decay", 1e-5, 1e-1, log=True)
    else:
        grad_decay = None

    name = f"trial_{trial.number:04d}"
    L.seed_everything(1234 + trial.number, workers=True)
    t0 = time.time()

    vm = ValueNetwork(D, bias=bias_val)
    model = OnPolicyValueLive(
        base_score_module=base_drift, value_module=vm, reward_function=reward_fn,
        dim=D, a=a, lr=lr, loss_type="quad",
        grad_decay=grad_decay, ema_decay=ema_decay,
    )
    smc_value = make_smc_value(smc_type, k, l, model)
    ds = OnPolicySMCDataset(
        dim=D, drift=base_drift, value=model.value_module,
        smc_value=smc_value, reward=reward_fn, device=DEVICE, a=a,
        batch_size=DS_BATCH, n_steps=n_steps, mc_samples_per_step=mc_samples,
        sampling_method=method, lambda_eff=lambda_eff,
        off_policy_frac=off_policy_frac, include_t_zero=False,
        random_t=random_t, generating_function=gmm_sample,
    )
    loader = DataLoader(ds, batch_size=BS)
    logger = CSVLogger(LOG_DIR, name=name, version=0)
    trainer = L.Trainer(
        max_steps=MAX_STEPS,
        val_check_interval=max(1, MAX_STEPS // N_VAL),
        callbacks=[OptunaPruning(trial)],
        logger=logger, enable_checkpointing=False,
        enable_progress_bar=False,
    )
    csv_path = f"{LOG_DIR}/{name}/version_0/metrics.csv"
    status = "complete"
    try:
        trainer.fit(model, loader, val_dataloaders=val_loader)
    except optuna.TrialPruned:
        status = "pruned"
        raise
    except RuntimeError as e:
        # NaN/inf loss or similar bad config — steer TPE away, don't crash.
        status = f"error:{type(e).__name__}"
        del model, vm, trainer, loader, ds
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"  {name}: {status} ({e}) -> LCB=-100", flush=True)
        return -100.0
    finally:
        if status == "pruned":
            print(f"  {name}: pruned  ({(time.time()-t0)/60:.1f} min)", flush=True)

    curve = _read_val_curve(csv_path)
    del model, vm, trainer, loader, ds
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if len(curve) < 5:
        return -100.0
    tail = curve[-LCB_TAIL:]
    n = len(tail)
    xx = np.arange(n, dtype=float)
    # OLS detrend: σ from residuals about the linear fit (don't penalise slope)
    A = np.vstack([xx, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, tail, rcond=None)
    resid = tail - A @ coef
    dof = max(1, n - 2)
    sigma = float(np.sqrt((resid ** 2).sum() / dof))
    sem = sigma / np.sqrt(n)
    lcb = float(tail.mean() - LCB_Z * sem)
    print(f"  {name}: complete  LCB={lcb:.3f}  "
          f"(tail mean={tail.mean():.3f} σ_detr={sigma:.3f} sem={sem:.3f} "
          f"n_tail={n}, n_val={len(curve)})  [{method}, smc={smc_type}, "
          f"mc={mc_samples}, nstep={n_steps}, rt={random_t}, "
          f"ofp={off_policy_frac:.2f}, lr={lr:.1e}]  "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    return lcb


def main():
    os.makedirs("notebooks", exist_ok=True)
    sampler = TPESampler(multivariate=True, group=True, seed=42)
    pruner = HyperbandPruner(min_resource=500, max_resource=MAX_STEPS,
                             reduction_factor=3)
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STUDY_DB, load_if_exists=True,
        direction="maximize", sampler=sampler, pruner=pruner,
    )
    n_done = len([t for t in study.trials
                  if t.state.is_finished()])
    remaining = max(0, N_TRIALS - n_done)
    print(f"Study '{STUDY_NAME}'  device={DEVICE}  "
          f"already-finished={n_done}  remaining={remaining}", flush=True)

    t_start = time.time()

    def _cb(study, trial):
        done = len([t for t in study.trials if t.state.is_finished()])
        try:
            best = study.best_value
            bt = study.best_trial.number
        except ValueError:
            best, bt = float("nan"), -1
        print(f"[{done}/{N_TRIALS}] elapsed={(time.time()-t_start)/60:.1f} min "
              f"| best LCB={best:.3f} (trial {bt})", flush=True)

    study.optimize(objective, n_trials=remaining, callbacks=[_cb],
                   gc_after_trial=True)

    print("\n" + "=" * 78)
    print("BEST TRIAL")
    print("=" * 78)
    bt = study.best_trial
    print(f"  number = {bt.number}")
    print(f"  LCB    = {bt.value:.4f}")
    for kk, vv in sorted(bt.params.items()):
        print(f"    {kk:>16} = {vv}")

    # param importances (best-effort)
    try:
        imp = optuna.importance.get_param_importances(study)
        print("\n  Param importances:")
        for kk, vv in imp.items():
            print(f"    {kk:>16} = {vv:.3f}")
    except Exception as e:
        imp = {}
        print(f"  (importance failed: {e})")

    summary = {
        "study_name": STUDY_NAME,
        "n_trials_total": len(study.trials),
        "n_complete": len([t for t in study.trials
                           if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials
                         if t.state == optuna.trial.TrialState.PRUNED]),
        "best": {"number": bt.number, "lcb": bt.value, "params": bt.params},
        "param_importances": imp,
        "top10": sorted(
            [{"number": t.number, "lcb": t.value, "params": t.params}
             for t in study.trials
             if t.state == optuna.trial.TrialState.COMPLETE
             and t.value is not None],
            key=lambda d: d["lcb"], reverse=True)[:10],
    }
    with open("notebooks/optuna_onpolicy_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: notebooks/optuna_onpolicy_results.json")
    print(f"Total elapsed: {(time.time()-t_start)/60:.1f} min")
    print("Done.")


if __name__ == "__main__":
    main()
