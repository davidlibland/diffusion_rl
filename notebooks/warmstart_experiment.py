"""
Warm-start experiment: off-policy pre-training → on-policy fine-tuning.

Best on-policy method from prior experiments: anc_mctd_smc_model_lam0.8

Configurations (all with same total budget of 6000 steps):
  - pure_offpolicy:    6000 off-policy steps (baseline)
  - pure_onpolicy:     6000 on-policy steps (baseline)
  - warmstart_500:     500 off-policy → 5500 on-policy
  - warmstart_1000:    1000 off-policy → 5000 on-policy
  - warmstart_2000:    2000 off-policy → 4000 on-policy
  - warmstart_3000:    3000 off-policy → 3000 on-policy
"""

import json
import os
import time

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

# ---------------------------------------------------------------------------
# GMM Setup
# ---------------------------------------------------------------------------
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

means_np = clf.means_
sigmas_np = np.sqrt(clf.covariances_)
weights_np = clf.weights_


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
        m = self.means.double()
        s2 = self.sigma2.double()
        w = self.weights.double()
        dk = t_ * s2[None, :] + 2 * self.a * (1 - t_)
        marg_mean = t_[:, :, None] * m[None, :, :]
        diff2 = ((x[:, None, :] - marg_mean) ** 2).sum(-1)
        t_safe = t_ + 1e-40
        log_gauss = (
            -self.D / 2.0 * torch.log(2 * torch.pi * t_safe * dk)
            - diff2 / (2 * t_safe * dk)
        )
        log_w = torch.log(w)[None, :]
        log_pw = log_w + log_gauss
        log_pw = log_pw - torch.logsumexp(log_pw, dim=1, keepdim=True)
        tV = 2 * self.a * (1 - t_) * s2[None, :] / dk
        tmu = (
            s2[None, :, None] * x[:, None, :]
            + 2 * self.a * (1 - t_)[:, :, None] * m[None, :, :]
        ) / dk[:, :, None]
        log_zk = self._log_Z(tmu, tV)
        return torch.logsumexp(log_pw + log_zk, dim=1).float()


_anal_vm_cpu = AnalyticalValue(_means, _sigma2, _weights, a=a, c=c, D=D)


def anal_fn(x, t):
    return _anal_vm_cpu(x.cpu(), t.cpu()).to(x.device)


# Drift
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


def gmm_sample(n):
    k = np.random.choice(len(weights_np), size=n, p=weights_np)
    return means_np[k] + sigmas_np[k, np.newaxis] * np.random.randn(n, D)


with open("notebooks/analytical_target.json") as f:
    _anal_tgt = json.load(f)
E_OPT = _anal_tgt["E_opt"]
V_0_0 = _anal_tgt["V_0_0"]

all_rewards = reward(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()

print(f"E_OPT = {E_OPT:.4f}  V(0,0) = {V_0_0:.4f}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LR = 3e-3
LOADER_BATCH_SIZE = 256
TOTAL_STEPS = 6000
LOG_DIR = "lightning_logs/warmstart"
CKPT_DIR = "checkpoints/warmstart"

WARMSTART_LENGTHS = [0, 500, 1000, 2000, 3000]
# 0 = pure on-policy, others = off-policy for N steps then on-policy

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------
def train_offpolicy_phase(vm, max_steps, run_name):
    """Train off-policy and return the updated value module."""
    ds = InterpolatingNumpyDataset(
        generating_function=gmm_sample, a=a, batch_size=1024,
    )
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)

    model = OffPolicyValue(
        base_score_module=base_drift,
        reward_function=reward,
        value_module=vm,
        dim=D, a=a, lr=LR,
        loss_type="quad",
        analytical_value_fn=anal_fn,
    )

    val_interval = max(1, max_steps // 60)
    logger = CSVLogger(LOG_DIR, name=run_name, version=0)

    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=val_interval,
        callbacks=[],
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    trainer.fit(model, loader, val_dataloaders=val_loader)
    return vm  # modified in-place


def train_onpolicy_phase(vm, max_steps, run_name, start_step=0):
    """Train on-policy (anc_mctd_smc_model_lam0.8) and return updated vm."""
    smc_fn = vm  # self-consistent

    ds = OnPolicySMCDataset(
        dim=D, drift=base_drift, value=vm, smc_value=smc_fn,
        reward=reward, device=DEVICE, a=a,
        batch_size=32, n_steps=100, mc_samples_per_step=10,
        sampling_method="ancestral_mc_td_lambda",
        lambda_eff=0.8,
    )
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)

    model = OnPolicyValue(
        base_score_module=base_drift,
        value_module=vm,
        reward_function=reward,
        dim=D, a=a, lr=LR,
        loss_type="quad",
        analytical_value_fn=anal_fn,
    )

    val_interval = max(1, max_steps // 60)

    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{run_name}",
        save_last=True, save_top_k=1,
        monitor="val_reward_mean", mode="max",
        filename="best",
    )

    logger = CSVLogger(LOG_DIR, name=run_name,
                       version=1 if start_step > 0 else 0)

    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=val_interval,
        callbacks=[ckpt_cb],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    trainer.fit(model, loader, val_dataloaders=val_loader)
    return vm


# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------

# Pure off-policy baseline (full budget)
run_name = "pure_offpolicy"
print(f"\n{'='*70}")
print(f"  {run_name} ({TOTAL_STEPS} steps)")
print(f"{'='*70}")

csv_check = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
if os.path.exists(csv_check):
    df = pd.read_csv(csv_check)
    val = df.dropna(subset=["val_reward_mean"])
    if len(val) > 0 and val["step"].max() >= TOTAL_STEPS - 1:
        print("  Already complete, skipping.")
    else:
        os.remove(csv_check)

if not os.path.exists(csv_check):
    vm = ValueNetwork(D, bias=bias_val)
    t0 = time.perf_counter()
    train_offpolicy_phase(vm, TOTAL_STEPS, run_name)
    elapsed = time.perf_counter() - t0
    print(f"  Elapsed: {elapsed/60:.1f} min")


# Pure on-policy baseline (full budget)
run_name = "pure_onpolicy"
print(f"\n{'='*70}")
print(f"  {run_name} ({TOTAL_STEPS} steps)")
print(f"{'='*70}")

csv_check = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
if os.path.exists(csv_check):
    df = pd.read_csv(csv_check)
    val = df.dropna(subset=["val_reward_mean"])
    if len(val) > 0 and val["step"].max() >= TOTAL_STEPS - 1:
        print("  Already complete, skipping.")
    else:
        os.remove(csv_check)

if not os.path.exists(csv_check):
    vm = ValueNetwork(D, bias=bias_val)
    t0 = time.perf_counter()
    train_onpolicy_phase(vm, TOTAL_STEPS, run_name, start_step=0)
    elapsed = time.perf_counter() - t0
    print(f"  Elapsed: {elapsed/60:.1f} min")


# Warm-start runs
for ws_steps in WARMSTART_LENGTHS:
    if ws_steps == 0:
        continue  # already covered by pure_onpolicy

    on_steps = TOTAL_STEPS - ws_steps
    run_name = f"warmstart_{ws_steps}"

    print(f"\n{'='*70}")
    print(f"  {run_name} ({ws_steps} off-policy → {on_steps} on-policy)")
    print(f"{'='*70}")

    # Check if on-policy phase is already done
    csv_on = f"{LOG_DIR}/{run_name}/version_1/metrics.csv"
    if os.path.exists(csv_on):
        df = pd.read_csv(csv_on)
        val = df.dropna(subset=["val_reward_mean"])
        if len(val) > 0 and val["step"].max() >= on_steps - 1:
            print("  Already complete, skipping.")
            continue

    # Phase 1: off-policy warm-start
    csv_off = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
    vm = ValueNetwork(D, bias=bias_val)

    skip_offpolicy = False
    if os.path.exists(csv_off):
        df = pd.read_csv(csv_off)
        trn = df.dropna(subset=["train_loss"])
        if len(trn) > 0 and trn["step"].max() >= ws_steps - 1:
            print(f"  Off-policy phase already done, loading checkpoint...")
            # Need to retrain since we don't save off-policy checkpoints
            skip_offpolicy = False  # must retrain to get the weights

    print(f"  Phase 1: Off-policy for {ws_steps} steps...")

    # Clean up stale logs
    import shutil
    for v in [0, 1]:
        p = f"{LOG_DIR}/{run_name}/version_{v}"
        if os.path.exists(p):
            shutil.rmtree(p)

    t0 = time.perf_counter()
    train_offpolicy_phase(vm, ws_steps, run_name)
    elapsed_off = time.perf_counter() - t0
    print(f"  Phase 1 done: {elapsed_off/60:.1f} min")

    # Phase 2: on-policy fine-tuning (same vm, weights carry over)
    print(f"  Phase 2: On-policy for {on_steps} steps...")
    t0 = time.perf_counter()
    train_onpolicy_phase(vm, on_steps, run_name, start_step=ws_steps)
    elapsed_on = time.perf_counter() - t0
    print(f"  Phase 2 done: {elapsed_on/60:.1f} min")


# ---------------------------------------------------------------------------
# Load and plot results
# ---------------------------------------------------------------------------
print(f"\n\n{'='*70}")
print(f"  RESULTS")
print(f"{'='*70}")


def load_combined_metrics(run_name, ws_steps=0):
    """Load metrics, combining off-policy and on-policy phases."""
    if run_name == "pure_offpolicy":
        csv = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
        if not os.path.exists(csv):
            return None
        return pd.read_csv(csv)

    if run_name == "pure_onpolicy":
        csv = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
        if not os.path.exists(csv):
            return None
        return pd.read_csv(csv)

    # Warm-start: combine phase 0 (off-policy) and phase 1 (on-policy)
    csv_off = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
    csv_on = f"{LOG_DIR}/{run_name}/version_1/metrics.csv"

    dfs = []
    if os.path.exists(csv_off):
        df_off = pd.read_csv(csv_off)
        dfs.append(df_off)

    if os.path.exists(csv_on):
        df_on = pd.read_csv(csv_on)
        # Offset steps by warm-start length
        df_on = df_on.copy()
        df_on["step"] = df_on["step"] + ws_steps
        dfs.append(df_on)

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


# Summary table
all_runs = [
    ("pure_offpolicy", 0),
    ("pure_onpolicy", 0),
    ("warmstart_500", 500),
    ("warmstart_1000", 1000),
    ("warmstart_2000", 2000),
    ("warmstart_3000", 3000),
]

print(f"\n  {'Run':<25} {'Best Reward':>12} {'Final Reward':>13} {'Gap':>8}")
print(f"  {'-'*62}")

run_data = {}
for run_name, ws in all_runs:
    df = load_combined_metrics(run_name, ws)
    if df is None:
        print(f"  {run_name:<25} {'N/A':>12}")
        continue
    val = df.dropna(subset=["val_reward_mean"])
    if len(val) == 0:
        print(f"  {run_name:<25} {'N/A':>12}")
        continue
    best = val["val_reward_mean"].max()
    final = val["val_reward_mean"].iloc[-1]
    gap = E_OPT - best
    print(f"  {run_name:<25} {best:>12.4f} {final:>13.4f} {gap:>8.4f}")
    run_data[run_name] = (val["step"].values, val["val_reward_mean"].values, ws)


# ---------------------------------------------------------------------------
# Plot: Terminal reward vs steps
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_title("Warm-Start Experiment: Terminal Reward vs Training Steps", fontsize=14, fontweight="bold")

colors = {
    "pure_offpolicy": "black",
    "pure_onpolicy": "gray",
    "warmstart_500": plt.cm.viridis(0.2),
    "warmstart_1000": plt.cm.viridis(0.4),
    "warmstart_2000": plt.cm.viridis(0.6),
    "warmstart_3000": plt.cm.viridis(0.8),
}

linestyles = {
    "pure_offpolicy": "--",
    "pure_onpolicy": ":",
}

for run_name, ws in all_runs:
    if run_name not in run_data:
        continue
    steps, rewards, ws_len = run_data[run_name]
    ls = linestyles.get(run_name, "-")
    lw = 2.5 if "pure" in run_name else 1.5
    label = run_name.replace("_", " ").replace("warmstart", "warm-start")
    ax.plot(steps, rewards, color=colors[run_name], linestyle=ls, linewidth=lw, label=label)

    # Mark the transition point for warm-start runs
    if ws > 0:
        ax.axvline(ws, color=colors[run_name], linestyle=":", alpha=0.3, linewidth=0.8)

ax.axhline(E_OPT, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label=f"E_opt = {E_OPT:.3f}")
ax.set_xlabel("Training Steps (total)", fontsize=12)
ax.set_ylabel("Avg Terminal Reward", fontsize=12)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("notebooks/warmstart_reward.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: notebooks/warmstart_reward.png")
plt.close()


# ---------------------------------------------------------------------------
# Plot: Bias per t-bin (final checkpoint)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Value Function Bias at End of Training", fontsize=14, fontweight="bold")

bin_names = ["t00_20", "t20_40", "t40_60", "t60_80", "t80_100"]
bin_centers = [0.1, 0.3, 0.5, 0.7, 0.9]

for ax_idx, metric_prefix in enumerate(["val_bias", "val_mae"]):
    ax = axes[ax_idx]
    metric_label = "Bias" if "bias" in metric_prefix else "MAE"
    ax.set_title(f"Final {metric_label} vs Analytical V (per t-bin)", fontsize=12)

    for run_name, ws in all_runs:
        df = load_combined_metrics(run_name, ws)
        if df is None:
            continue

        cols = [f"{metric_prefix}_{b}" for b in bin_names]
        available = [c for c in cols if c in df.columns]
        if not available:
            continue

        # Get last row with these metrics
        sub = df.dropna(subset=available[:1])
        if len(sub) == 0:
            continue
        last_row = sub.iloc[-1]

        vals = [last_row.get(c, np.nan) for c in cols]
        ls = linestyles.get(run_name, "-")
        label = run_name.replace("_", " ").replace("warmstart", "ws")
        ax.plot(bin_centers, vals, "o-", color=colors[run_name], linestyle=ls,
                linewidth=1.5, markersize=5, label=label)

    if "bias" in metric_prefix:
        ax.axhline(0, color="black", linestyle=":", linewidth=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel(metric_label)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("notebooks/warmstart_bias.png", dpi=150, bbox_inches="tight")
print("Saved: notebooks/warmstart_bias.png")
plt.close()


print(f"\nDone. E_OPT = {E_OPT:.4f}")
