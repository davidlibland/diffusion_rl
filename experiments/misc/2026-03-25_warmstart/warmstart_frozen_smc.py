"""
Warm-start with frozen SMC: off-policy pre-training, then on-policy
fine-tuning where smc_value is frozen at the warm-start checkpoint
but value (for bootstrapping) continues to train.

This decouples the two feedback loops:
  - SMC resampling uses a FIXED model → stable sampling distribution
  - TD/bootstrap targets use the LIVE model → can still improve

Configurations:
  - pure_offpolicy:          6000 off-policy (baseline, from prior experiment)
  - frozen_smc_ws1000:       1000 off → 5000 on (smc=frozen)
  - frozen_smc_ws2000:       2000 off → 4000 on (smc=frozen)
  - frozen_smc_ws3000:       3000 off → 3000 on (smc=frozen)
  - live_smc_ws2000:         2000 off → 4000 on (smc=live, for comparison)
"""

import copy
import json
import os
import shutil
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
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

# ---------------------------------------------------------------------------
# GMM Setup (standard boilerplate)
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
            + (-10.0 * (m ** 2).sum(-1) + 20.0 * (m * cc).sum(-1)
               + 200.0 * v * (cc ** 2).sum()) / denom
            - 10.0 * (cc ** 2).sum()
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


with open("experiments/common/analytical_target.json") as f:
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
LOG_DIR = "lightning_logs/frozen_smc"
CKPT_DIR = "checkpoints/frozen_smc"

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_offpolicy(vm, max_steps, run_name):
    """Train off-policy, return updated vm."""
    ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=1024)
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
    model = OffPolicyValue(
        base_score_module=base_drift, reward_function=reward,
        value_module=vm, dim=D, a=a, lr=LR,
        loss_type="quad", analytical_value_fn=anal_fn,
    )
    val_interval = max(1, max_steps // 60)
    logger = CSVLogger(LOG_DIR, name=run_name, version=0)
    trainer = L.Trainer(
        max_steps=max_steps, val_check_interval=val_interval,
        logger=logger, enable_checkpointing=False, enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    return vm


def train_onpolicy(vm, smc_fn, max_steps, run_name, version=1):
    """Train on-policy with given smc_fn (frozen or live)."""
    ds = OnPolicySMCDataset(
        dim=D, drift=base_drift, value=vm, smc_value=smc_fn,
        reward=reward, device=DEVICE, a=a,
        batch_size=32, n_steps=100, mc_samples_per_step=10,
        sampling_method="ancestral_mc_td_lambda", lambda_eff=0.8,
    )
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
    model = OnPolicyValue(
        base_score_module=base_drift, value_module=vm,
        reward_function=reward, dim=D, a=a, lr=LR,
        loss_type="quad", analytical_value_fn=anal_fn,
    )
    val_interval = max(1, max_steps // 60)
    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{run_name}", save_last=True, save_top_k=1,
        monitor="val_reward_mean", mode="max", filename="best",
    )
    logger = CSVLogger(LOG_DIR, name=run_name, version=version)
    trainer = L.Trainer(
        max_steps=max_steps, val_check_interval=val_interval,
        callbacks=[ckpt_cb], logger=logger,
        enable_checkpointing=True, enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    return vm


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
RUNS = [
    # (name, ws_steps, frozen_smc)
    ("pure_offpolicy", TOTAL_STEPS, None),       # all off-policy
    ("frozen_smc_ws1000", 1000, True),
    ("frozen_smc_ws2000", 2000, True),
    ("frozen_smc_ws3000", 3000, True),
    ("live_smc_ws2000", 2000, False),             # comparison: live smc
]

for run_name, ws_steps, frozen_smc in RUNS:
    on_steps = TOTAL_STEPS - ws_steps if frozen_smc is not None else 0

    print(f"\n{'='*70}")
    if frozen_smc is None:
        print(f"  {run_name}: {ws_steps} off-policy steps")
    else:
        smc_label = "FROZEN" if frozen_smc else "LIVE"
        print(f"  {run_name}: {ws_steps} off → {on_steps} on (smc={smc_label})")
    print(f"{'='*70}")

    # Check if already complete
    if frozen_smc is None:
        csv_check = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
    else:
        csv_check = f"{LOG_DIR}/{run_name}/version_1/metrics.csv"

    if os.path.exists(csv_check):
        df = pd.read_csv(csv_check)
        val = df.dropna(subset=["val_reward_mean"])
        expected = TOTAL_STEPS if frozen_smc is None else on_steps
        if len(val) > 0 and val["step"].max() >= expected - 1:
            print("  Already complete, skipping.")
            continue

    # Clean up stale logs
    for v in range(3):
        p = f"{LOG_DIR}/{run_name}/version_{v}"
        if os.path.exists(p):
            shutil.rmtree(p)

    # Phase 1: off-policy
    vm = ValueNetwork(D, bias=bias_val)
    print(f"  Phase 1: Off-policy for {ws_steps} steps...")
    t0 = time.perf_counter()
    train_offpolicy(vm, ws_steps, run_name)
    elapsed = time.perf_counter() - t0
    print(f"  Phase 1 done: {elapsed/60:.1f} min")

    if frozen_smc is None:
        continue  # pure off-policy, no phase 2

    # Create frozen SMC copy BEFORE on-policy training mutates vm
    if frozen_smc:
        smc_fn = copy.deepcopy(vm).to(DEVICE)
        smc_fn.eval()
        for p in smc_fn.parameters():
            p.requires_grad_(False)
        print(f"  Created frozen SMC copy of warm-started model (on {DEVICE}).")
    else:
        smc_fn = vm  # live: same object, mutates with training

    # Phase 2: on-policy
    print(f"  Phase 2: On-policy for {on_steps} steps (smc={'frozen' if frozen_smc else 'live'})...")
    t0 = time.perf_counter()
    train_onpolicy(vm, smc_fn, on_steps, run_name, version=1)
    elapsed = time.perf_counter() - t0
    print(f"  Phase 2 done: {elapsed/60:.1f} min")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\n\n{'='*70}")
print(f"  RESULTS (E_OPT = {E_OPT:.4f})")
print(f"{'='*70}")


def load_combined(run_name, ws_steps, has_onpolicy):
    dfs = []
    csv_off = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
    if os.path.exists(csv_off):
        dfs.append(pd.read_csv(csv_off))

    if has_onpolicy:
        csv_on = f"{LOG_DIR}/{run_name}/version_1/metrics.csv"
        if os.path.exists(csv_on):
            df_on = pd.read_csv(csv_on).copy()
            df_on["step"] = df_on["step"] + ws_steps
            dfs.append(df_on)

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


all_configs = [
    ("pure_offpolicy", TOTAL_STEPS, False),
    ("frozen_smc_ws1000", 1000, True),
    ("frozen_smc_ws2000", 2000, True),
    ("frozen_smc_ws3000", 3000, True),
    ("live_smc_ws2000", 2000, True),
]

print(f"\n  {'Run':<28} {'Best Reward':>12} {'Final Reward':>13} {'Gap':>8}")
print(f"  {'-'*65}")

run_data = {}
for run_name, ws, has_on in all_configs:
    df = load_combined(run_name, ws, has_on)
    if df is None:
        print(f"  {run_name:<28} {'N/A':>12}")
        continue
    val = df.dropna(subset=["val_reward_mean"])
    if len(val) == 0:
        print(f"  {run_name:<28} {'N/A':>12}")
        continue
    best = val["val_reward_mean"].max()
    final = val["val_reward_mean"].iloc[-1]
    gap = E_OPT - best
    print(f"  {run_name:<28} {best:>12.4f} {final:>13.4f} {gap:>8.4f}")
    run_data[run_name] = (val["step"].values, val["val_reward_mean"].values, ws)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_title("Frozen vs Live SMC: Terminal Reward vs Training Steps",
             fontsize=14, fontweight="bold")

colors = {
    "pure_offpolicy": "black",
    "frozen_smc_ws1000": plt.cm.cool(0.2),
    "frozen_smc_ws2000": plt.cm.cool(0.5),
    "frozen_smc_ws3000": plt.cm.cool(0.8),
    "live_smc_ws2000": "red",
}
linestyles = {
    "pure_offpolicy": "--",
    "live_smc_ws2000": ":",
}

for run_name, ws, has_on in all_configs:
    if run_name not in run_data:
        continue
    steps, rewards, ws_len = run_data[run_name]
    ls = linestyles.get(run_name, "-")
    lw = 2.5 if "pure" in run_name else 1.5
    label = run_name.replace("_", " ").replace("frozen smc ", "frozen-smc ").replace("live smc ", "live-smc ")
    ax.plot(steps, rewards, color=colors[run_name], linestyle=ls, linewidth=lw, label=label)

    if has_on:
        ax.axvline(ws_len, color=colors[run_name], linestyle=":", alpha=0.3, linewidth=0.8)

ax.axhline(E_OPT, color="red", linestyle=":", linewidth=1.5, alpha=0.7,
           label=f"E_opt = {E_OPT:.3f}")
ax.set_xlabel("Training Steps (total)", fontsize=12)
ax.set_ylabel("Avg Terminal Reward", fontsize=12)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("experiments/misc/2026-03-25_warmstart/frozen_smc_reward.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: experiments/misc/2026-03-25_warmstart/frozen_smc_reward.png")
plt.close()

print(f"\nDone.")
