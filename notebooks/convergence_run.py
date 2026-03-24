"""
Run single_seed_TD(λ=0.6) and single_seed_MC to convergence.

From LR sweep: best lr=3e-3 for both; estimated convergence steps:
  TD(λ=0.6): ~8827 steps (~17 min)
  MC:        ~3210 steps (~6 min)

Runs with generous headroom (2x estimate) and reports when gap < 0.5.
"""

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
# Setup
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
LR = 3e-3

reward = lambda x: -10 * (x - torch.tensor([[1.0, 0.0]]).to(x)).square().sum(dim=1)
all_rewards = reward(torch.from_numpy(X).float())
max_reward  = all_rewards.max()
bias = (torch.log(torch.mean(torch.exp(all_rewards - max_reward))) + max_reward).item()

base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

E_OPT = json.loads(open("notebooks/analytical_target.json").read())["E_opt"]
LOG_DIR = "lightning_logs/convergence_run"

# From LR sweep: tau fits → use 2x the estimated convergence steps as budget
RUNS = {
    "single_seed_td_lam0.6": dict(
        sampling_method="single_seed_td_lambda", lambda_eff=0.6,
        max_steps=13000,   # 1.5× ~8827  (~28 min)
    ),
    "single_seed_mc": dict(
        sampling_method="single_seed_mc", lambda_eff=0.1,
        max_steps=5000,    # 1.5× ~3210  (~11 min)
    ),
}

# ---------------------------------------------------------------------------
# Calibrate
# ---------------------------------------------------------------------------
print("Calibrating steps/sec...")
CALIBRATION_STEPS = 20
_cv = ValueNetwork(DIM, bias=bias)
_ds = OnPolicySMCDataset(
    dim=DIM, drift=base_drift, value=_cv, smc_value=_cv, reward=reward,
    device=DEVICE, a=a, batch_size=32, n_steps=100,
    sampling_method="single_seed_mc",
)
_loader  = DataLoader(_ds, batch_size=BATCH_SIZE)
_model   = OnPolicyValue(base_score_module=base_drift, value_module=_cv, dim=DIM, a=a, lr=LR)
_trainer = L.Trainer(max_steps=CALIBRATION_STEPS, enable_checkpointing=False,
                     enable_progress_bar=False, logger=False)
t0 = time.perf_counter()
_trainer.fit(_model, _loader)
elapsed = time.perf_counter() - t0
steps_per_sec = CALIBRATION_STEPS / elapsed
print(f"  {steps_per_sec:.2f} steps/s")


# ---------------------------------------------------------------------------
# Run each method
# ---------------------------------------------------------------------------
def build(sampling_method, lambda_eff, max_steps):
    vm = ValueNetwork(DIM, bias=bias)
    ds = OnPolicySMCDataset(
        dim=DIM, drift=base_drift, value=vm, smc_value=vm, reward=reward,
        device=DEVICE, a=a, batch_size=32, n_steps=100,
        sampling_method=sampling_method, lambda_eff=lambda_eff,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    model  = OnPolicyValue(
        base_score_module=base_drift, value_module=vm,
        reward_function=reward, dim=DIM, a=a, lr=LR,
        loss_type="quad",
    )
    return model, loader


for run_name, cfg in RUNS.items():
    max_steps = cfg["max_steps"]
    est_min   = max_steps / steps_per_sec / 60
    print(f"\n{'='*60}")
    print(f"{run_name}  (max_steps={max_steps}, est {est_min:.1f} min)")
    print(f"{'='*60}")

    model, loader = build(**cfg)
    val_interval  = max(1, max_steps // 40)   # ~40 val points

    ckpt_dir = f"checkpoints/convergence_run/{run_name}"
    ckpt_cb  = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
        every_n_train_steps=max(1, max_steps // 5),
        save_top_k=1,
        monitor="val_reward_mean",
        mode="max",
        filename="best",
    )

    logger  = CSVLogger(LOG_DIR, name=run_name, version=0)
    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=val_interval,
        callbacks=[ckpt_cb],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    t0 = time.perf_counter()
    trainer.fit(model, loader, val_dataloaders=val_loader)
    elapsed = time.perf_counter() - t0

    # Report
    csv_path = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
    df  = pd.read_csv(csv_path)
    val = df.dropna(subset=["val_reward_mean"])
    steps   = val["step"].values
    rewards = val["val_reward_mean"].values
    best_r  = rewards.max()
    final_r = rewards[-1]

    print(f"\n  Elapsed: {elapsed/60:.1f} min")
    print(f"  Best val_reward_mean:  {best_r:.4f}  (gap {E_OPT - best_r:.4f})")
    print(f"  Final val_reward_mean: {final_r:.4f}  (gap {E_OPT - final_r:.4f})")
    print(f"  Optimal target:        {E_OPT:.4f}")

    # Print full curve
    print(f"\n  {'step':>8}  {'reward':>10}  {'gap':>8}")
    for s, r in zip(steps, rewards):
        marker = "  ← best" if r == best_r else ""
        print(f"  {int(s):>8}  {r:>10.4f}  {E_OPT - r:>8.4f}{marker}")


print(f"\nDone. Optimal target E_opt = {E_OPT:.4f}")
