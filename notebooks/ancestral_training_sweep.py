# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Ancestral Training Sweep: SMC Value & Lambda Comparison
#
# Compare training dynamics of **ancestral_td_lambda** and **ancestral_mc_td_lambda**
# with `smc_value=reward` vs `smc_value=model`, across 5 lambda values,
# against the off-policy baseline.
#
# All runs are controlled for **total training samples seen** (not steps).

# %%
import json
import os
import time
from math import ceil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import reduce
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

# %% [markdown]
# ## Setup

# %%
# GMM setup (standard boilerplate)
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler()
X = scalar.fit_transform(X)
clf = GaussianMixture(n_components=100, covariance_type="spherical")
clf.fit(X)

_means = torch.from_numpy(clf.means_).double()
_sigma2 = torch.from_numpy(clf.covariances_).double()
_weights = torch.from_numpy(clf.weights_).double()
_sigmas = torch.sqrt(_sigma2)[:, None]
_weights_col = _weights[:, None]

D = 2
a = 1.0
c = torch.tensor([1.0, 0.0])

# %%
# Analytical value function
class AnalyticalValue(nn.Module):
    def __init__(self, means, sigma2, weights, a=1.0, c=None, D=2):
        super().__init__()
        if c is None:
            c = torch.tensor([1.0, 0.0])
        self.register_buffer("means", means.float())
        self.register_buffer("sigma2", sigma2.float())
        self.register_buffer("weights", weights.float())
        self.a = a
        self.D = D
        self.register_buffer("c", c.float())

    def _log_Z(self, m, v):
        c = self.c.double()
        denom = 1.0 + 20.0 * v
        return (
            -self.D / 2.0 * torch.log(denom)
            + (-10.0 * (m ** 2).sum(-1) + 20.0 * (m * c).sum(-1)
               + 200.0 * v * (c ** 2).sum()) / denom
            - 10.0 * (c ** 2).sum()
        )

    def forward(self, x, t):
        x = x.double()
        t = t.double().reshape(-1)
        if t.numel() == 1:
            t = t.expand(x.shape[0])
        t_ = t[:, None]
        means = self.means.double()
        sigma2 = self.sigma2.double()
        weights = self.weights.double()
        eps = 1e-40
        dk = t_ * sigma2[None, :] + 2 * self.a * (1 - t_)
        marg_mean = t_[:, :, None] * means[None, :, :]
        diff2 = ((x[:, None, :] - marg_mean) ** 2).sum(-1)
        t_safe = t_ + eps
        log_gauss = (
            -self.D / 2.0 * torch.log(2 * torch.pi * t_safe * dk)
            - diff2 / (2 * t_safe * dk)
        )
        log_w = torch.log(weights)[None, :]
        log_pw = log_w + log_gauss
        log_pw = log_pw - torch.logsumexp(log_pw, dim=1, keepdim=True)
        tV = 2 * self.a * (1 - t_) * sigma2[None, :] / dk
        tmu = (
            sigma2[None, :, None] * x[:, None, :]
            + 2 * self.a * (1 - t_)[:, :, None] * means[None, :, :]
        ) / dk[:, :, None]
        log_zk = self._log_Z(tmu, tV)
        return torch.logsumexp(log_pw + log_zk, dim=1).float()


_anal_vm_cpu = AnalyticalValue(_means, _sigma2, _weights, a=a, c=c, D=D)


def anal_fn(x, t):
    result = _anal_vm_cpu(x.cpu(), t.cpu())
    return result.to(x.device)


# %%
# Drift, reward, SMC functions
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
    new_means = (2 * a * (1 - ts_) * means_ + xt_ * sigmas_[None, ...] ** 2) / denominator[:, None, :]
    new_sigmas = torch.sqrt(0.5 * torch.exp(log_std_factor) * sigmas_ ** 2)
    return {"log_weights": log_weights, "means": new_means, "sigmas": new_sigmas}


def gmm_drift(xt, ts, a):
    ts = ts.reshape(-1, 1)
    cond = get_conditional_mixture(
        xt, ts,
        _means.float().to(xt), _sigmas.float().to(xt), _weights_col.float().to(xt), a
    )
    new_weights = torch.exp(cond["log_weights"])
    us = (cond["means"] - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(new_weights[:, None, :] * us, "n d m -> n d", "sum")


DEVICE = "mps"
base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)
reward = lambda x: -10 * (x - c.to(x)).square().sum(dim=1)

smc_const = lambda x, t: torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
smc_reward = lambda x, t: reward(x)

# GMM sampler for off-policy
means_np = clf.means_
sigmas_np = np.sqrt(clf.covariances_)
weights_np = clf.weights_


def gmm_sample(n):
    k = np.random.choice(len(weights_np), size=n, p=weights_np)
    return means_np[k] + sigmas_np[k, np.newaxis] * np.random.randn(n, D)


# %%
# Load analytical targets
with open("notebooks/analytical_target.json") as f:
    _anal_tgt = json.load(f)
E_OPT = _anal_tgt["E_opt"]
V_0_0 = _anal_tgt["V_0_0"]
print(f"E_OPT = {E_OPT:.4f}  V(0,0) = {V_0_0:.4f}")

all_rewards = reward(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()

# %% [markdown]
# ## Experiment Configuration
#
# **Key design choices:**
# - **Sample-controlled comparison**: All methods see the same total number of training
#   samples. For ancestral methods, one dataset call produces
#   `batch_size * mc_samples * (n_steps-1)` samples. We adjust `max_steps` so that
#   `max_steps * loader_batch_size` is constant across all runs.
# - **Lambda values**: 5 values spanning the TD-MC spectrum: 0.0 (pure TD), 0.2, 0.5, 0.8, 1.0 (pure MC)
# - **Two ancestral methods**: `ancestral_td_lambda` and `ancestral_mc_td_lambda`
# - **Two SMC strategies**: `smc_value=reward` and `smc_value=model` (self-consistent)
# - **Off-policy baseline**: Standard off-policy training with same sample budget

# %%
# Experiment parameters
LAMBDA_VALUES = [0.0, 0.2, 0.5, 0.8, 1.0]
LAMBDA_LABELS = ["0.0", "0.2", "0.5", "0.8", "1.0"]

LR = 3e-3
LOADER_BATCH_SIZE = 256
DATASET_BATCH_SIZE = 32
N_STEPS = 100
MC_SAMPLES = 10

# Total sample budget: we want roughly the same number of gradient updates
# for all methods. Each gradient step processes LOADER_BATCH_SIZE samples.
# We'll do ~3000 gradient steps for early-to-mid dynamics.
MAX_STEPS = 3000
VAL_INTERVAL = max(1, MAX_STEPS // 60)  # ~60 validation checkpoints

LOG_DIR = "lightning_logs/ancestral_sweep"
CKPT_DIR = "checkpoints/ancestral_sweep"

# %% [markdown]
# ## Training Functions

# %%
class SampleCounter(Callback):
    """Track cumulative training samples seen."""
    def __init__(self, samples_per_step):
        super().__init__()
        self.samples_per_step = samples_per_step
        self.total_samples = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.total_samples += self.samples_per_step


def build_onpolicy(sampling_method, lambda_eff, smc_mode, max_steps, run_name):
    """Build on-policy model + dataset + trainer.

    Args:
        sampling_method: "ancestral_td_lambda" or "ancestral_mc_td_lambda"
        lambda_eff: effective lambda in [0, 1]
        smc_mode: "reward" or "model"
        max_steps: number of gradient steps
        run_name: name for logging
    """
    vm = ValueNetwork(D, bias=bias_val)

    if smc_mode == "model":
        smc_fn = vm  # self-consistent: smc_value = model
    else:
        smc_fn = smc_reward

    ds = OnPolicySMCDataset(
        dim=D,
        drift=base_drift,
        value=vm,
        smc_value=smc_fn,
        reward=reward,
        device=DEVICE,
        a=a,
        batch_size=DATASET_BATCH_SIZE,
        n_steps=N_STEPS,
        mc_samples_per_step=MC_SAMPLES,
        sampling_method=sampling_method,
        lambda_eff=lambda_eff,
    )
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)

    model = OnPolicyValue(
        base_score_module=base_drift,
        value_module=vm,
        reward_function=reward,
        dim=D,
        a=a,
        lr=LR,
        loss_type="quad",
        analytical_value_fn=anal_fn,
    )

    val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)
    val_interval = max(1, max_steps // 60)

    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{run_name}",
        save_last=True,
        save_top_k=1,
        monitor="val_reward_mean",
        mode="max",
        filename="best",
    )

    sample_counter = SampleCounter(LOADER_BATCH_SIZE)

    logger = CSVLogger(LOG_DIR, name=run_name, version=0)
    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=val_interval,
        callbacks=[ckpt_cb, sample_counter],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    return model, loader, val_loader, trainer, sample_counter


def build_offpolicy(max_steps, run_name):
    """Build off-policy model + dataset + trainer."""
    vm = ValueNetwork(D, bias=bias_val)

    ds = InterpolatingNumpyDataset(
        generating_function=gmm_sample,
        a=a,
        batch_size=1024,
    )
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)

    model = OffPolicyValue(
        base_score_module=base_drift,
        reward_function=reward,
        value_module=vm,
        dim=D,
        a=a,
        lr=LR,
        loss_type="quad",
        analytical_value_fn=anal_fn,
    )

    val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)
    val_interval = max(1, max_steps // 60)

    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{run_name}",
        save_last=True,
        save_top_k=1,
        monitor="val_reward_mean",
        mode="max",
        filename="best",
    )

    sample_counter = SampleCounter(LOADER_BATCH_SIZE)

    logger = CSVLogger(LOG_DIR, name=run_name, version=0)
    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=val_interval,
        callbacks=[ckpt_cb, sample_counter],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    return model, loader, val_loader, trainer, sample_counter


# %% [markdown]
# ## Run All Experiments

# %%
# Define all runs
RUNS = {}

# Off-policy baseline
RUNS["offpolicy"] = dict(
    build_fn="offpolicy",
    max_steps=MAX_STEPS,
)

# Ancestral TD(lambda) with smc=reward
for i, lam in enumerate(LAMBDA_VALUES):
    name = f"anc_td_smc_reward_lam{LAMBDA_LABELS[i]}"
    RUNS[name] = dict(
        build_fn="onpolicy",
        sampling_method="ancestral_td_lambda",
        lambda_eff=lam,
        smc_mode="reward",
        max_steps=MAX_STEPS,
    )

# Ancestral TD(lambda) with smc=model (self-consistent)
for i, lam in enumerate(LAMBDA_VALUES):
    name = f"anc_td_smc_model_lam{LAMBDA_LABELS[i]}"
    RUNS[name] = dict(
        build_fn="onpolicy",
        sampling_method="ancestral_td_lambda",
        lambda_eff=lam,
        smc_mode="model",
        max_steps=MAX_STEPS,
    )

# Ancestral MC-TD(lambda) with smc=reward
for i, lam in enumerate(LAMBDA_VALUES):
    name = f"anc_mctd_smc_reward_lam{LAMBDA_LABELS[i]}"
    RUNS[name] = dict(
        build_fn="onpolicy",
        sampling_method="ancestral_mc_td_lambda",
        lambda_eff=lam,
        smc_mode="reward",
        max_steps=MAX_STEPS,
    )

# Ancestral MC-TD(lambda) with smc=model (self-consistent)
for i, lam in enumerate(LAMBDA_VALUES):
    name = f"anc_mctd_smc_model_lam{LAMBDA_LABELS[i]}"
    RUNS[name] = dict(
        build_fn="onpolicy",
        sampling_method="ancestral_mc_td_lambda",
        lambda_eff=lam,
        smc_mode="model",
        max_steps=MAX_STEPS,
    )

print(f"Total runs: {len(RUNS)}")
for name in RUNS:
    print(f"  {name}")

# %%
# Execute all runs sequentially
results = {}

for run_name, cfg in RUNS.items():
    max_steps = cfg["max_steps"]
    print(f"\n{'='*70}")
    print(f"  {run_name}  (max_steps={max_steps})")
    print(f"{'='*70}")

    # Skip if already run (check for metrics CSV)
    csv_path = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
    if os.path.exists(csv_path):
        print(f"  Already run, loading from {csv_path}")
        results[run_name] = {"csv_path": csv_path}
        continue

    if cfg["build_fn"] == "offpolicy":
        model, loader, val_loader, trainer, counter = build_offpolicy(
            max_steps, run_name
        )
    else:
        model, loader, val_loader, trainer, counter = build_onpolicy(
            sampling_method=cfg["sampling_method"],
            lambda_eff=cfg["lambda_eff"],
            smc_mode=cfg["smc_mode"],
            max_steps=max_steps,
            run_name=run_name,
        )

    t0 = time.perf_counter()
    trainer.fit(model, loader, val_dataloaders=val_loader)
    elapsed = time.perf_counter() - t0

    print(f"  Elapsed: {elapsed/60:.1f} min, samples: {counter.total_samples:,}")
    results[run_name] = {
        "csv_path": csv_path,
        "elapsed_s": elapsed,
        "total_samples": counter.total_samples,
    }

print("\n\nAll runs complete!")

# %% [markdown]
# ## Load & Plot Results

# %%
def load_metrics(run_name):
    """Load metrics CSV for a run."""
    csv_path = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df


def extract_val_series(df, metric="val_reward_mean"):
    """Extract validation metric as (step, value) arrays."""
    val = df.dropna(subset=[metric])
    if len(val) == 0:
        return np.array([]), np.array([])
    return val["step"].values, val[metric].values


def extract_val_bias_variance(df):
    """Extract per-bin MAE and bias from validation logs."""
    bin_names = ["t00_20", "t20_40", "t40_60", "t60_80", "t80_100"]

    # Compute average MAE across bins
    mae_cols = [f"val_mae_{b}" for b in bin_names]
    bias_cols = [f"val_bias_{b}" for b in bin_names]

    available_mae = [c for c in mae_cols if c in df.columns]
    available_bias = [c for c in bias_cols if c in df.columns]

    if not available_mae:
        return np.array([]), np.array([]), np.array([])

    # MAE rows
    mae_df = df.dropna(subset=available_mae[:1])
    if len(mae_df) == 0:
        return np.array([]), np.array([]), np.array([])

    steps = mae_df["step"].values
    avg_mae = mae_df[available_mae].mean(axis=1).values

    if available_bias:
        bias_df = df.dropna(subset=available_bias[:1])
        avg_bias = bias_df[available_bias].mean(axis=1).values
    else:
        avg_bias = np.full_like(avg_mae, np.nan)

    return steps, avg_mae, avg_bias


# %%
# Samples per step: for all methods with LOADER_BATCH_SIZE=256,
# each training step processes 256 samples.
SAMPLES_PER_STEP = LOADER_BATCH_SIZE


def steps_to_samples(steps):
    """Convert training steps to total samples seen."""
    return steps * SAMPLES_PER_STEP


# %%
# Color and style definitions
METHOD_STYLES = {
    "offpolicy": dict(color="black", linestyle="--", linewidth=2, label="Off-Policy"),
}

# Generate colors for lambda values
_cmap = plt.cm.viridis
_lambda_colors = [_cmap(i / (len(LAMBDA_VALUES) - 1)) for i in range(len(LAMBDA_VALUES))]

# %% [markdown]
# ### Plot 1: Terminal Reward vs Samples Seen

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
fig.suptitle("Terminal Reward vs Training Samples", fontsize=16, fontweight="bold")

panel_configs = [
    ("Ancestral TD(λ), smc=reward", "anc_td_smc_reward"),
    ("Ancestral TD(λ), smc=model", "anc_td_smc_model"),
    ("Ancestral MC-TD(λ), smc=reward", "anc_mctd_smc_reward"),
    ("Ancestral MC-TD(λ), smc=model", "anc_mctd_smc_model"),
]

for ax, (title, prefix) in zip(axes.flat, panel_configs):
    ax.set_title(title, fontsize=13)

    # Off-policy baseline
    df_off = load_metrics("offpolicy")
    if df_off is not None:
        steps, vals = extract_val_series(df_off, "val_reward_mean")
        if len(steps) > 0:
            ax.plot(steps_to_samples(steps), vals,
                    color="black", linestyle="--", linewidth=2, label="Off-Policy", alpha=0.7)

    # Lambda sweep
    for i, (lam, lam_label) in enumerate(zip(LAMBDA_VALUES, LAMBDA_LABELS)):
        run_name = f"{prefix}_lam{lam_label}"
        df = load_metrics(run_name)
        if df is None:
            continue
        steps, vals = extract_val_series(df, "val_reward_mean")
        if len(steps) > 0:
            ax.plot(steps_to_samples(steps), vals,
                    color=_lambda_colors[i], linewidth=1.5,
                    label=f"λ={lam_label}")

    # Optimal reward line
    ax.axhline(E_OPT, color="red", linestyle=":", linewidth=1, alpha=0.7, label=f"E_opt={E_OPT:.3f}")

    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Avg Terminal Reward")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("notebooks/ancestral_sweep_reward.png", dpi=150, bbox_inches="tight")
print("Saved: notebooks/ancestral_sweep_reward.png")
plt.close()

# %% [markdown]
# ### Plot 2: Model Bias (V_model - V_analytical) vs Samples Seen

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
fig.suptitle("Avg Model Bias (V_model - V_analytical) vs Training Samples", fontsize=16, fontweight="bold")

for ax, (title, prefix) in zip(axes.flat, panel_configs):
    ax.set_title(title, fontsize=13)

    # Off-policy baseline
    df_off = load_metrics("offpolicy")
    if df_off is not None:
        steps, _, avg_bias = extract_val_bias_variance(df_off)
        if len(steps) > 0:
            ax.plot(steps_to_samples(steps), avg_bias,
                    color="black", linestyle="--", linewidth=2, label="Off-Policy", alpha=0.7)

    # Lambda sweep
    for i, (lam, lam_label) in enumerate(zip(LAMBDA_VALUES, LAMBDA_LABELS)):
        run_name = f"{prefix}_lam{lam_label}"
        df = load_metrics(run_name)
        if df is None:
            continue
        steps, _, avg_bias = extract_val_bias_variance(df)
        if len(steps) > 0:
            ax.plot(steps_to_samples(steps), avg_bias,
                    color=_lambda_colors[i], linewidth=1.5,
                    label=f"λ={lam_label}")

    ax.axhline(0, color="red", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Avg Bias")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("notebooks/ancestral_sweep_bias.png", dpi=150, bbox_inches="tight")
print("Saved: notebooks/ancestral_sweep_bias.png")
plt.close()

# %% [markdown]
# ### Plot 3: Model MAE (|V_model - V_analytical|) vs Samples Seen

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
fig.suptitle("Avg MAE |V_model - V_analytical| vs Training Samples", fontsize=16, fontweight="bold")

for ax, (title, prefix) in zip(axes.flat, panel_configs):
    ax.set_title(title, fontsize=13)

    # Off-policy baseline
    df_off = load_metrics("offpolicy")
    if df_off is not None:
        steps, avg_mae, _ = extract_val_bias_variance(df_off)
        if len(steps) > 0:
            ax.plot(steps_to_samples(steps), avg_mae,
                    color="black", linestyle="--", linewidth=2, label="Off-Policy", alpha=0.7)

    # Lambda sweep
    for i, (lam, lam_label) in enumerate(zip(LAMBDA_VALUES, LAMBDA_LABELS)):
        run_name = f"{prefix}_lam{lam_label}"
        df = load_metrics(run_name)
        if df is None:
            continue
        steps, avg_mae, _ = extract_val_bias_variance(df)
        if len(steps) > 0:
            ax.plot(steps_to_samples(steps), avg_mae,
                    color=_lambda_colors[i], linewidth=1.5,
                    label=f"λ={lam_label}")

    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Avg MAE")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

plt.tight_layout()
plt.savefig("notebooks/ancestral_sweep_mae.png", dpi=150, bbox_inches="tight")
print("Saved: notebooks/ancestral_sweep_mae.png")
plt.close()

# %% [markdown]
# ### Plot 4: Summary - Final Reward by Method/Lambda/SMC

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Final Terminal Reward by Configuration", fontsize=14, fontweight="bold")

for ax_idx, (smc_label, smc_key) in enumerate([("smc=reward", "smc_reward"), ("smc=model", "smc_model")]):
    ax = axes[ax_idx]
    ax.set_title(smc_label, fontsize=13)

    # Collect final rewards
    for method_label, method_prefix in [("Ancestral TD(λ)", "anc_td"), ("Ancestral MC-TD(λ)", "anc_mctd")]:
        final_rewards = []
        for lam_label in LAMBDA_LABELS:
            run_name = f"{method_prefix}_{smc_key}_lam{lam_label}"
            df = load_metrics(run_name)
            if df is not None:
                steps, vals = extract_val_series(df, "val_reward_mean")
                if len(vals) > 0:
                    final_rewards.append(vals[-1])
                else:
                    final_rewards.append(np.nan)
            else:
                final_rewards.append(np.nan)

        ax.plot(LAMBDA_VALUES, final_rewards, "o-", linewidth=2, markersize=8, label=method_label)

    # Off-policy baseline
    df_off = load_metrics("offpolicy")
    if df_off is not None:
        _, vals = extract_val_series(df_off, "val_reward_mean")
        if len(vals) > 0:
            ax.axhline(vals[-1], color="black", linestyle="--", linewidth=1.5,
                       label=f"Off-Policy (final={vals[-1]:.3f})", alpha=0.7)

    ax.axhline(E_OPT, color="red", linestyle=":", linewidth=1, alpha=0.7,
               label=f"E_opt={E_OPT:.3f}")

    ax.set_xlabel("λ")
    ax.set_ylabel("Final Avg Terminal Reward")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("notebooks/ancestral_sweep_summary.png", dpi=150, bbox_inches="tight")
print("Saved: notebooks/ancestral_sweep_summary.png")
plt.close()

# %% [markdown]
# ### Plot 5: Training Loss vs Samples

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
fig.suptitle("Training Loss vs Samples", fontsize=16, fontweight="bold")

for ax, (title, prefix) in zip(axes.flat, panel_configs):
    ax.set_title(title, fontsize=13)

    # Off-policy baseline
    df_off = load_metrics("offpolicy")
    if df_off is not None:
        train = df_off.dropna(subset=["train_loss"])
        if len(train) > 0:
            # Subsample for readability
            step_idx = np.linspace(0, len(train) - 1, min(200, len(train)), dtype=int)
            ax.plot(steps_to_samples(train["step"].values[step_idx]),
                    train["train_loss"].values[step_idx],
                    color="black", linestyle="--", linewidth=1, label="Off-Policy", alpha=0.5)

    for i, (lam, lam_label) in enumerate(zip(LAMBDA_VALUES, LAMBDA_LABELS)):
        run_name = f"{prefix}_lam{lam_label}"
        df = load_metrics(run_name)
        if df is None:
            continue
        train = df.dropna(subset=["train_loss"])
        if len(train) > 0:
            step_idx = np.linspace(0, len(train) - 1, min(200, len(train)), dtype=int)
            ax.plot(steps_to_samples(train["step"].values[step_idx]),
                    train["train_loss"].values[step_idx],
                    color=_lambda_colors[i], linewidth=1, alpha=0.7,
                    label=f"λ={lam_label}")

    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

plt.tight_layout()
plt.savefig("notebooks/ancestral_sweep_loss.png", dpi=150, bbox_inches="tight")
print("Saved: notebooks/ancestral_sweep_loss.png")
plt.close()

# %% [markdown]
# ### Save Summary JSON

# %%
summary = {
    "E_OPT": E_OPT,
    "V_0_0": V_0_0,
    "max_steps": MAX_STEPS,
    "loader_batch_size": LOADER_BATCH_SIZE,
    "samples_per_step": SAMPLES_PER_STEP,
    "total_sample_budget": MAX_STEPS * SAMPLES_PER_STEP,
    "lambda_values": LAMBDA_VALUES,
    "runs": {},
}

for run_name in RUNS:
    df = load_metrics(run_name)
    if df is None:
        continue
    steps, vals = extract_val_series(df, "val_reward_mean")
    steps_mae, avg_mae, avg_bias = extract_val_bias_variance(df)

    entry = {
        "final_reward": float(vals[-1]) if len(vals) > 0 else None,
        "best_reward": float(vals.max()) if len(vals) > 0 else None,
        "best_reward_step": int(steps[vals.argmax()]) if len(vals) > 0 else None,
        "final_mae": float(avg_mae[-1]) if len(avg_mae) > 0 else None,
        "final_bias": float(avg_bias[-1]) if len(avg_bias) > 0 else None,
    }
    summary["runs"][run_name] = entry

with open("notebooks/ancestral_sweep_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved: notebooks/ancestral_sweep_results.json")

# %%
# Print summary table
print(f"\n{'='*90}")
print(f"  SUMMARY (E_OPT = {E_OPT:.4f}, total samples = {MAX_STEPS * SAMPLES_PER_STEP:,})")
print(f"{'='*90}")
print(f"  {'Run':<42} {'Final Reward':>14} {'Best Reward':>14} {'Gap':>8} {'MAE':>8}")
print(f"  {'-'*86}")
for run_name, entry in summary["runs"].items():
    fr = entry["final_reward"]
    br = entry["best_reward"]
    mae = entry["final_mae"]
    gap = E_OPT - br if br is not None else None
    print(f"  {run_name:<42} {fr:>14.4f} {br:>14.4f} {gap:>8.4f} {mae:>8.4f}" if fr is not None else f"  {run_name:<42} {'N/A':>14}")

print(f"\nDone! All plots saved to notebooks/ancestral_sweep_*.png")
