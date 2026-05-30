"""
Test one_step_bootstrap training with 3 SMC modes:
  - smc=const (uniform resampling)
  - smc=reward
  - smc=model (self-consistent)

All warm-started from 2000 steps of off-policy, then 4000 steps on-policy.
Also test with frozen SMC (best idea from prior experiment).
"""

import copy
import json
import os
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import reduce
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
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
        cc = self.c.double()
        denom = 1.0 + 20.0 * v
        return (
            -self.D / 2.0 * torch.log(denom)
            + (
                -10.0 * (m**2).sum(-1)
                + 20.0 * (m * cc).sum(-1)
                + 200.0 * v * (cc**2).sum()
            )
            / denom
            - 10.0 * (cc**2).sum()
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
        mm = t_[:, :, None] * m[None, :, :]
        d2 = ((x[:, None, :] - mm) ** 2).sum(-1)
        lg = -self.D / 2.0 * torch.log(2 * torch.pi * (t_ + 1e-40) * dk) - d2 / (
            2 * (t_ + 1e-40) * dk
        )
        lw = torch.log(w)[None, :]
        lpw = lw + lg - torch.logsumexp(lw + lg, dim=1, keepdim=True)
        tV = 2 * self.a * (1 - t_) * s2[None, :] / dk
        tmu = (
            s2[None, :, None] * x[:, None, :]
            + 2 * self.a * (1 - t_)[:, :, None] * m[None, :, :]
        ) / dk[:, :, None]
        return torch.logsumexp(lpw + self._log_Z(tmu, tV), dim=1).float()


_anal_vm_cpu = AnalyticalValue(_means, _sigma2, _weights, a=a, c=c, D=D)


def anal_fn(x, t):
    return _anal_vm_cpu(x.cpu(), t.cpu()).to(x.device)


def get_conditional_mixture(xt, ts, means, sigmas, weights, a):
    n, d = xt.shape
    xt_ = xt[..., None]
    means_ = means.T[None, ...]
    ts_ = ts[..., None]
    sigmas_ = sigmas.T
    weights_ = weights.T
    olw = torch.log(weights_)
    denom = 2 * a * (1 - ts) + ts * sigmas_**2
    le = -reduce((xt_ - means_ * ts_) ** 2, "n d m -> n m", "sum") / (2 * ts * denom)
    lsf = torch.log(2 * a * (1 - ts) / denom) * d / 2
    lrw = olw + le + lsf
    lw = lrw - torch.logsumexp(lrw, dim=1, keepdim=True)
    lw = torch.where((ts == 0), olw, lw)
    nm = (2 * a * (1 - ts_) * means_ + xt_ * sigmas_[None, ...] ** 2) / denom[
        :, None, :
    ]
    return {"log_weights": lw, "means": nm}


def gmm_drift(xt, ts, a):
    ts = ts.reshape(-1, 1)
    cond = get_conditional_mixture(
        xt,
        ts,
        _means.float().to(xt),
        _sigmas.float().to(xt),
        _weights_col.float().to(xt),
        a,
    )
    us = (cond["means"] - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(
        torch.exp(cond["log_weights"])[:, None, :] * us, "n d m -> n d", "sum"
    )


DEVICE = "mps"
base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(
    dtype=torch.float
)
reward = lambda x: -10 * (x - c.to(x)).square().sum(dim=1)
smc_const = lambda x, t: torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
smc_reward = lambda x, t: reward(x)


def gmm_sample(n):
    k = np.random.choice(len(weights_np), size=n, p=weights_np)
    return means_np[k] + sigmas_np[k, np.newaxis] * np.random.randn(n, D)


with open("experiments/common/analytical_target.json") as f:
    _anal_tgt = json.load(f)
E_OPT = _anal_tgt["E_opt"]

all_rewards = reward(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()

print(f"E_OPT = {E_OPT:.4f}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LR = 3e-3
LOADER_BATCH_SIZE = 256
WS_STEPS = 0
ON_STEPS = 4000 + 2000
TOTAL_STEPS = WS_STEPS + ON_STEPS
LOG_DIR = "lightning_logs/one_step_bootstrap_no_warm"
CKPT_DIR = "checkpoints/one_step_bootstrap_no_warm"

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


def train_offpolicy(vm, max_steps, run_name):
    ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=1024)
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
    logger = CSVLogger(LOG_DIR, name=run_name, version=0)
    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=max(1, max_steps // 60),
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    return vm


def train_onpolicy(vm, smc_fn, max_steps, run_name, sampling_method, version=1):
    ds = OnPolicySMCDataset(
        dim=D,
        drift=base_drift,
        value=vm,
        smc_value=smc_fn,
        reward=reward,
        device=DEVICE,
        a=a,
        batch_size=32,
        n_steps=100,
        mc_samples_per_step=10,
        sampling_method=sampling_method,
        lambda_eff=0.0,
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
    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{run_name}",
        save_last=True,
        save_top_k=1,
        monitor="val_reward_mean",
        mode="max",
        filename="best",
    )
    logger = CSVLogger(LOG_DIR, name=run_name, version=version)
    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=max(1, max_steps // 60),
        callbacks=[ckpt_cb],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    return vm


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
RUNS = [
    # (name, smc_mode, frozen_smc)
    ("osb_smc_const", "const", False),
    ("osb_smc_reward", "reward", False),
    ("osb_smc_model", "model", False),
    ("osb_smc_model_frozen", "model", True),  # frozen SMC at warm-start
    (
        "osb_smc_reward_frozen",
        "reward",
        True,
    ),  # frozen reward (same as live, but for symmetry)
]

# Also run pure off-policy baseline
run_name = "pure_offpolicy"
print(f"\n{'=' * 70}")
print(f"  {run_name}: {TOTAL_STEPS} off-policy steps")
print(f"{'=' * 70}")

csv_check = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
if os.path.exists(csv_check):
    df = pd.read_csv(csv_check)
    val = df.dropna(subset=["val_reward_mean"])
    if len(val) > 0 and val["step"].max() >= TOTAL_STEPS - 1:
        print("  Already complete, skipping.")
    else:
        shutil.rmtree(f"{LOG_DIR}/{run_name}", ignore_errors=True)
        vm = ValueNetwork(D, bias=bias_val)
        train_offpolicy(vm, TOTAL_STEPS, run_name)
else:
    vm = ValueNetwork(D, bias=bias_val)
    train_offpolicy(vm, TOTAL_STEPS, run_name)


for run_name, smc_mode, frozen_smc in RUNS:
    print(f"\n{'=' * 70}")
    smc_label = f"{smc_mode}{'_frozen' if frozen_smc else ''}"
    print(
        f"  {run_name}: {WS_STEPS} off → {ON_STEPS} on (one_step_bootstrap, smc={smc_label})"
    )
    print(f"{'=' * 70}")

    csv_on = f"{LOG_DIR}/{run_name}/version_1/metrics.csv"
    if os.path.exists(csv_on):
        df = pd.read_csv(csv_on)
        val = df.dropna(subset=["val_reward_mean"])
        if len(val) > 0 and val["step"].max() >= ON_STEPS - 1:
            print("  Already complete, skipping.")
            continue

    # Clean stale
    for v in range(3):
        p = f"{LOG_DIR}/{run_name}/version_{v}"
        if os.path.exists(p):
            shutil.rmtree(p)

    # Phase 1: off-policy warm-start
    vm = ValueNetwork(D, bias=bias_val)
    print(f"  Phase 1: Off-policy for {WS_STEPS} steps...")
    t0 = time.perf_counter()
    train_offpolicy(vm, WS_STEPS, run_name)
    print(f"  Phase 1 done: {(time.perf_counter() - t0) / 60:.1f} min")

    # Set up SMC function
    if smc_mode == "const":
        smc_fn = smc_const
    elif smc_mode == "reward":
        smc_fn = smc_reward
    elif smc_mode == "model":
        if frozen_smc:
            smc_fn = copy.deepcopy(vm).to(DEVICE)
            smc_fn.eval()
            for p in smc_fn.parameters():
                p.requires_grad_(False)
            print(f"  Frozen SMC copy created (on {DEVICE}).")
        else:
            smc_fn = vm  # live, self-consistent

    # For reward+frozen: smc_reward is already stateless, same as live
    if smc_mode == "reward" and frozen_smc:
        smc_fn = smc_reward

    # Phase 2: on-policy with one_step_bootstrap
    print(f"  Phase 2: On-policy (one_step_bootstrap) for {ON_STEPS} steps...")
    t0 = time.perf_counter()
    train_onpolicy(vm, smc_fn, ON_STEPS, run_name, "one_step_bootstrap", version=1)
    print(f"  Phase 2 done: {(time.perf_counter() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\n\n{'=' * 70}")
print(f"  RESULTS (E_OPT = {E_OPT:.4f})")
print(f"{'=' * 70}")


def load_combined(run_name, ws_steps, has_on):
    dfs = []
    csv_off = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
    if os.path.exists(csv_off):
        dfs.append(pd.read_csv(csv_off))
    if has_on:
        csv_on = f"{LOG_DIR}/{run_name}/version_1/metrics.csv"
        if os.path.exists(csv_on):
            df_on = pd.read_csv(csv_on).copy()
            df_on["step"] = df_on["step"] + ws_steps
            dfs.append(df_on)
    return pd.concat(dfs, ignore_index=True) if dfs else None


all_configs = [
    ("pure_offpolicy", TOTAL_STEPS, False),
    ("osb_smc_const", WS_STEPS, True),
    ("osb_smc_reward", WS_STEPS, True),
    ("osb_smc_model", WS_STEPS, True),
    ("osb_smc_model_frozen", WS_STEPS, True),
    ("osb_smc_reward_frozen", WS_STEPS, True),
]

print(f"\n  {'Run':<30} {'Best Reward':>12} {'Final Reward':>13} {'Gap':>8}")
print(f"  {'-' * 67}")

run_data = {}
for rn, ws, has_on in all_configs:
    df = load_combined(rn, ws, has_on)
    if df is None:
        print(f"  {rn:<30} {'N/A':>12}")
        continue
    val = df.dropna(subset=["val_reward_mean"])
    if len(val) == 0:
        print(f"  {rn:<30} {'N/A':>12}")
        continue
    best = val["val_reward_mean"].max()
    final = val["val_reward_mean"].iloc[-1]
    gap = E_OPT - best
    print(f"  {rn:<30} {best:>12.4f} {final:>13.4f} {gap:>8.4f}")
    run_data[rn] = (val["step"].values, val["val_reward_mean"].values, ws)


# Plot
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_title(
    "One-Step Bootstrap: Terminal Reward vs Training Steps",
    fontsize=14,
    fontweight="bold",
)

colors = {
    "pure_offpolicy": "black",
    "osb_smc_const": "gray",
    "osb_smc_reward": "blue",
    "osb_smc_model": "green",
    "osb_smc_model_frozen": "purple",
    "osb_smc_reward_frozen": "cyan",
}
ls_map = {"pure_offpolicy": "--", "osb_smc_const": ":"}

for rn, ws, has_on in all_configs:
    if rn not in run_data:
        continue
    steps, rewards, ws_len = run_data[rn]
    ls = ls_map.get(rn, "-")
    lw = 2.5 if "pure" in rn else 1.5
    ax.plot(steps, rewards, color=colors[rn], linestyle=ls, linewidth=lw, label=rn)
    if has_on:
        ax.axvline(ws_len, color=colors[rn], linestyle=":", alpha=0.3, linewidth=0.8)

ax.axhline(
    E_OPT,
    color="red",
    linestyle=":",
    linewidth=1.5,
    alpha=0.7,
    label=f"E_opt={E_OPT:.3f}",
)
ax.set_xlabel("Training Steps (total)")
ax.set_ylabel("Avg Terminal Reward")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "experiments/misc/2026-03-25_one_step_bootstrap/one_step_bootstrap_reward_no_warm.png", dpi=150, bbox_inches="tight"
)
print("\nSaved: experiments/misc/2026-03-25_one_step_bootstrap/one_step_bootstrap_reward_no_warm.png")
plt.close()

print("\nDone.")
