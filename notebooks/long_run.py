"""
Long run for single_seed_MC and single_seed_TD(λ=0.6).

Phase 1: 30-minute run with checkpointing, dense val logging.
Phase 2: Re-estimate time to convergence from phase 1 curves.
Phase 3: Resume from checkpoint and run to the estimated convergence step.
"""

import glob
import json
import time

import numpy as np
import pandas as pd
import torch
from einops import reduce
from scipy.optimize import curve_fit
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

# ---------------------------------------------------------------------------
# Dataset / GMM (same as other scripts)
# ---------------------------------------------------------------------------
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler()
X = scalar.fit_transform(X)

clf = GaussianMixture(n_components=100, covariance_type="spherical")
clf.fit(X)

means   = torch.from_numpy(clf.means_)
sigmas  = torch.sqrt(torch.from_numpy(clf.covariances_))[:, None]
weights = torch.from_numpy(clf.weights_)[:, None]


def get_conditional_mixture(xt, ts, means, sigmas, weights, a):
    n, d = xt.shape
    xt_ = xt[..., None]
    means_ = means.T[None, ...]
    ts_ = ts[..., None]
    sigmas_ = sigmas.T
    weights_ = weights.T
    orig_log_weights = torch.log(weights_)
    denominator = 2 * a * (1 - ts) + ts * sigmas_ ** 2
    likelihood_exp_numerator = reduce((xt_ - means_ * ts_) ** 2, "n d m -> n m", "sum")
    likelihood_exp = -likelihood_exp_numerator / (2 * ts * denominator)
    log_std_factor = torch.log(2 * a * (1 - ts) / denominator) * d / 2
    log_rel_weights = orig_log_weights + likelihood_exp + log_std_factor
    normalization = torch.logsumexp(log_rel_weights, dim=1, keepdim=True)
    log_weights = log_rel_weights - normalization
    log_weights = torch.where((ts == 0), orig_log_weights, log_weights)
    std_factor = torch.exp(log_std_factor)
    new_means = (2 * a * (1 - ts_) * means_ + xt_ * sigmas_[None, ...] ** 2) / denominator[:, None, :]
    new_sigmas = torch.sqrt(0.5 * std_factor * sigmas_ ** 2)
    return {"log_weights": log_weights, "means": new_means, "sigmas": new_sigmas}


def gmm_drift(xt, ts, a):
    ts = ts.reshape(-1, 1)
    cond = get_conditional_mixture(xt, ts, means.to(xt), sigmas.to(xt), weights.to(xt), a)
    new_weights = torch.exp(cond["log_weights"])
    us = (cond["means"] - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(new_weights[:, None, :] * us, "n d m -> n d", "sum")


a = 1
DIM = 2
DEVICE = "mps"
BATCH_SIZE = 256

reward = lambda x: -10 * (x - torch.tensor([[1.0, 0.0]]).to(x)).square().sum(dim=1)
all_rewards = reward(torch.from_numpy(X).float())
max_reward  = all_rewards.max()
bias = (torch.log(torch.mean(torch.exp(all_rewards - max_reward))) + max_reward).item()

base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

E_OPT = json.loads(open("notebooks/analytical_target.json").read())["E_opt"]
LOG_DIR = "lightning_logs/long_run"

# ---------------------------------------------------------------------------
# Calibrate steps/sec using single_seed_mc (same as moons_sweep.py)
# ---------------------------------------------------------------------------
CALIBRATION_STEPS = 20
PHASE1_MINUTES = 30

print("Calibrating against single_seed_mc...")
_cv = ValueNetwork(DIM, bias=bias)
_calib_dataset = OnPolicySMCDataset(
    dim=DIM, drift=base_drift, value=_cv, smc_value=_cv, reward=reward,
    device=DEVICE, a=a, batch_size=32, n_steps=100,
    sampling_method="single_seed_mc",
)
_calib_loader = DataLoader(_calib_dataset, batch_size=BATCH_SIZE)
_calib_model  = OnPolicyValue(base_score_module=base_drift, value_module=_cv, dim=DIM, a=a, lr=1e-2)
_calib_trainer = L.Trainer(max_steps=CALIBRATION_STEPS, enable_checkpointing=False,
                            enable_progress_bar=False, logger=False)
t0 = time.perf_counter()
_calib_trainer.fit(_calib_model, _calib_loader)
elapsed = time.perf_counter() - t0

steps_per_sec = CALIBRATION_STEPS / elapsed
PHASE1_STEPS  = int(steps_per_sec * PHASE1_MINUTES * 60)
VAL_STEPS     = max(1, PHASE1_STEPS // 10)   # ~10 val points over 30 min
CKPT_STEPS    = max(1, PHASE1_STEPS // 5)    # checkpoint every 6 min

print(f"  {CALIBRATION_STEPS} steps in {elapsed:.1f}s → {steps_per_sec:.2f} steps/s")
print(f"  PHASE1_STEPS={PHASE1_STEPS}  VAL_STEPS={VAL_STEPS}  CKPT_STEPS={CKPT_STEPS}")


# ---------------------------------------------------------------------------
# Helper: build model + dataset for a given method
# ---------------------------------------------------------------------------
def build(sampling_method, lambda_eff=0.1):
    value_module = ValueNetwork(DIM, bias=bias)
    dataset = OnPolicySMCDataset(
        dim=DIM, drift=base_drift, value=value_module, smc_value=value_module,
        reward=reward, device=DEVICE, a=a, batch_size=32, n_steps=100,
        sampling_method=sampling_method, lambda_eff=lambda_eff,
    )
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    model = OnPolicyValue(
        base_score_module=base_drift, value_module=value_module,
        reward_function=reward, dim=DIM, a=a, lr=1e-2,
        loss_type="quad", grad_decay=1e-8,
    )
    return model, train_loader


RUNS = {
    "single_seed_mc":          dict(sampling_method="single_seed_mc",         lambda_eff=0.1),
    "single_seed_td_lam0.6":   dict(sampling_method="single_seed_td_lambda",  lambda_eff=0.6),
}

# ---------------------------------------------------------------------------
# Phase 1: 30-minute runs
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nPHASE 1: {PHASE1_MINUTES}-minute runs\n{'='*60}")

phase1_ckpt = {}

for run_name, cfg in RUNS.items():
    print(f"\n--- {run_name} ---")
    model, train_loader = build(**cfg)

    ckpt_dir = f"checkpoints/long_run/{run_name}"
    ckpt_cb  = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
        every_n_train_steps=CKPT_STEPS,
        save_top_k=1,
        monitor="val_reward_mean",
        mode="max",
        filename="best",
    )

    logger  = CSVLogger(LOG_DIR, name=run_name, version=0)
    trainer = L.Trainer(
        max_steps=PHASE1_STEPS,
        val_check_interval=VAL_STEPS,
        callbacks=[ckpt_cb],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, val_dataloaders=val_loader)
    last_ckpt = ckpt_cb.last_model_path
    phase1_ckpt[run_name] = last_ckpt
    print(f"  checkpoint: {last_ckpt}")


# ---------------------------------------------------------------------------
# Phase 2: fit convergence curve and estimate remaining steps
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nPHASE 2: convergence estimation\n{'='*60}")


def load_val_curve(log_dir, run_name, version=0):
    path = f"{log_dir}/{run_name}/version_{version}/metrics.csv"
    df   = pd.read_csv(path)
    val  = df.dropna(subset=["val_reward_mean"]).copy()
    return val


def estimate_convergence(steps, rewards, e_opt, target_gap=1.0):
    """
    Fit r(t) = E_opt - A*exp(-t/tau) and return the step where
    E_opt - r(t) < target_gap, i.e. t = tau * log(A / target_gap).
    Returns (tau, t_conv) or (nan, nan) if fit fails.
    """
    gap = e_opt - rewards
    valid = gap > 0
    if valid.sum() < 3:
        return float("nan"), float("nan")
    try:
        def exp_model(t, A, tau):
            return A * np.exp(-t / tau)
        popt, _ = curve_fit(
            exp_model, steps[valid], gap[valid],
            p0=[gap[valid][0], steps[valid][-1]],
            bounds=([0, 1], [np.inf, np.inf]),
            maxfev=5000,
        )
        A, tau = popt
        if A <= target_gap:
            return tau, 0.0
        t_conv = tau * np.log(A / target_gap)
        return tau, t_conv
    except Exception:
        return float("nan"), float("nan")


phase2_extra_steps = {}

for run_name in RUNS:
    val = load_val_curve(LOG_DIR, run_name, version=0)
    steps   = val["step"].values.astype(float)
    rewards = val["val_reward_mean"].values

    print(f"\n{run_name}:")
    for s, r in zip(steps, rewards):
        print(f"  step {s:6.0f}: {r:7.2f}  (gap {E_OPT - r:.2f})")

    tau, t_conv = estimate_convergence(steps, rewards, E_OPT, target_gap=1.0)
    current_max = steps[-1]
    extra = max(0.0, t_conv - current_max)

    if np.isnan(t_conv):
        print(f"  → fit failed; using 2× current as fallback")
        extra = current_max
    else:
        extra_min = extra / (steps_per_sec * 60)
        print(f"  → tau={tau:.0f} steps, t_conv={t_conv:.0f}, extra={extra:.0f} steps "
              f"({extra_min:.1f} min)")

    phase2_extra_steps[run_name] = int(extra)


# ---------------------------------------------------------------------------
# Phase 3: resume from checkpoint and run extra steps
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nPHASE 3: resume and run to convergence\n{'='*60}")

for run_name, cfg in RUNS.items():
    extra = phase2_extra_steps[run_name]
    if extra == 0:
        print(f"\n{run_name}: already converged, skipping.")
        continue

    ckpt_path = phase1_ckpt[run_name]
    total_steps = PHASE1_STEPS + extra
    extra_min   = extra / (steps_per_sec * 60)
    print(f"\n--- {run_name}: +{extra} steps ({extra_min:.1f} min) ---")
    print(f"  resuming from: {ckpt_path}")

    model, train_loader = build(**cfg)

    ckpt_dir = f"checkpoints/long_run/{run_name}"
    ckpt_cb  = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
        every_n_train_steps=CKPT_STEPS,
        save_top_k=1,
        monitor="val_reward_mean",
        mode="max",
        filename="best_p3",
    )

    # Append to the same CSV (version=0) so metrics are continuous
    logger = CSVLogger(LOG_DIR, name=run_name, version=0)
    val_steps_p3 = max(1, extra // 10)
    trainer = L.Trainer(
        max_steps=total_steps,
        val_check_interval=val_steps_p3,
        callbacks=[ckpt_cb],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    print(f"  done. checkpoint: {ckpt_cb.last_model_path}")


# ---------------------------------------------------------------------------
# Final results
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}")
for run_name in RUNS:
    val = load_val_curve(LOG_DIR, run_name, version=0)
    steps   = val["step"].values
    rewards = val["val_reward_mean"].values
    print(f"\n{run_name}:")
    for s, r in zip(steps, rewards):
        marker = " ← best" if r == rewards.max() else ""
        print(f"  step {s:6d}: {r:7.2f}  (gap {E_OPT - r:.2f}){marker}")
    print(f"  optimal target: {E_OPT:.2f}")
