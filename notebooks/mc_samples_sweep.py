"""
Sweep over mc_samples_per_step to assess impact on training performance.

Uses ancestral_td_lambda at λ=0 (≡ one_step_bootstrap) with smc=reward.
No warm-start. Varies mc_samples in {2, 5, 10, 20, 40}.
Controls for total samples seen: adjusts batch_size so that
batch_size * mc_samples is constant (≈ dataset batch_size).

Also runs off-policy baseline for comparison.
"""

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
from math import ceil
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
D = 2; a = 1.0; c = torch.tensor([1.0, 0.0])
means_np = clf.means_; sigmas_np = np.sqrt(clf.covariances_); weights_np = clf.weights_


class AnalyticalValue(nn.Module):
    def __init__(self, means, sigma2, weights, a=1.0, c=None, D=2):
        super().__init__()
        if c is None: c = torch.tensor([1.0, 0.0])
        self.register_buffer("means", means.float())
        self.register_buffer("sigma2", sigma2.float())
        self.register_buffer("weights", weights.float())
        self.a = a; self.D = D; self.register_buffer("c", c.float())
    def _log_Z(self, m, v):
        cc = self.c.double(); denom = 1.0 + 20.0 * v
        return (-self.D/2.0*torch.log(denom) + (-10*(m**2).sum(-1)+20*(m*cc).sum(-1)+200*v*(cc**2).sum())/denom - 10*(cc**2).sum())
    def forward(self, x, t):
        x = x.double(); t = t.double().reshape(-1)
        if t.numel() == 1: t = t.expand(x.shape[0])
        t_ = t[:, None]; m = self.means.double(); s2 = self.sigma2.double(); w = self.weights.double()
        dk = t_*s2[None,:]+2*self.a*(1-t_); mm = t_[:,:,None]*m[None,:,:]
        d2 = ((x[:,None,:]-mm)**2).sum(-1); ts = t_+1e-40
        lg = (-self.D/2.0*torch.log(2*torch.pi*ts*dk)-d2/(2*ts*dk))
        lw = torch.log(w)[None,:]; lpw = lw+lg-torch.logsumexp(lw+lg,dim=1,keepdim=True)
        tV = 2*self.a*(1-t_)*s2[None,:]/dk
        tmu = (s2[None,:,None]*x[:,None,:]+2*self.a*(1-t_)[:,:,None]*m[None,:,:])/dk[:,:,None]
        return torch.logsumexp(lpw+self._log_Z(tmu,tV),dim=1).float()

_anal_vm_cpu = AnalyticalValue(_means, _sigma2, _weights, a=a, c=c, D=D)
def anal_fn(x, t): return _anal_vm_cpu(x.cpu(), t.cpu()).to(x.device)

def get_conditional_mixture(xt, ts, means, sigmas, weights, a):
    n, d = xt.shape; xt_ = xt[...,None]; means_ = means.T[None,...]; ts_ = ts[...,None]
    sigmas_ = sigmas.T; weights_ = weights.T; olw = torch.log(weights_)
    denom = 2*a*(1-ts)+ts*sigmas_**2
    le = -reduce((xt_-means_*ts_)**2,"n d m -> n m","sum")/(2*ts*denom)
    lsf = torch.log(2*a*(1-ts)/denom)*d/2
    lrw = olw+le+lsf; lw = lrw-torch.logsumexp(lrw,dim=1,keepdim=True)
    lw = torch.where((ts==0),olw,lw)
    nm = (2*a*(1-ts_)*means_+xt_*sigmas_[None,...]**2)/denom[:,None,:]
    return {"log_weights": lw, "means": nm}

def gmm_drift(xt, ts, a):
    ts = ts.reshape(-1,1)
    cond = get_conditional_mixture(xt,ts,_means.float().to(xt),_sigmas.float().to(xt),_weights_col.float().to(xt),a)
    us = (cond["means"]-xt[:,:,None])/(1-ts[...,None])
    return reduce(torch.exp(cond["log_weights"])[:,None,:]*us,"n d m -> n d","sum")

DEVICE = "mps"
base_drift = lambda x, t: gmm_drift(x, t if t.ndim>=1 else t.unsqueeze(0), a).to(dtype=torch.float)
reward = lambda x: -10*(x-c.to(x)).square().sum(dim=1)
smc_reward = lambda x, t: reward(x)
def gmm_sample(n):
    k = np.random.choice(len(weights_np),size=n,p=weights_np)
    return means_np[k]+sigmas_np[k,np.newaxis]*np.random.randn(n,D)

with open("notebooks/analytical_target.json") as f: _at = json.load(f)
E_OPT = _at["E_opt"]
all_rewards = reward(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards-max_r)))+max_r).item()
print(f"E_OPT = {E_OPT:.4f}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LR = 3e-3
LOADER_BATCH_SIZE = 256
MAX_STEPS = 6000
LOG_DIR = "lightning_logs/mc_samples_sweep"
CKPT_DIR = "checkpoints/mc_samples_sweep"

# Target: dataset_batch_size * mc_samples ≈ 320 particles per call
# (with n_steps=100, each call produces ~320 * 101 ≈ 32320 samples)
TARGET_PARTICLES = 320
MC_VALUES = [2, 5, 10, 20, 40]

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

RUNS = []

# Off-policy baseline
RUNS.append(dict(name="offpolicy", mc=None, ds_batch=None))

# mc_samples sweep
for mc in MC_VALUES:
    ds_batch = max(1, TARGET_PARTICLES // mc)
    RUNS.append(dict(name=f"mc{mc}", mc=mc, ds_batch=ds_batch))

print(f"Runs:")
for r in RUNS:
    if r["mc"] is None:
        print(f"  {r['name']}: off-policy baseline")
    else:
        total = r["ds_batch"] * r["mc"]
        print(f"  {r['name']}: mc_samples={r['mc']}, ds_batch={r['ds_batch']}, particles/call={total}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
for cfg in RUNS:
    run_name = cfg["name"]
    print(f"\n{'='*70}")
    print(f"  {run_name}")
    print(f"{'='*70}")

    csv_path = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        val = df.dropna(subset=["val_reward_mean"])
        if len(val) > 0 and val["step"].max() >= MAX_STEPS - 1:
            print("  Already complete, skipping.")
            continue

    for v in range(3):
        p = f"{LOG_DIR}/{run_name}/version_{v}"
        if os.path.exists(p): shutil.rmtree(p)

    vm = ValueNetwork(D, bias=bias_val)

    if cfg["mc"] is None:
        # Off-policy
        ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=1024)
        loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
        model = OffPolicyValue(
            base_score_module=base_drift, reward_function=reward,
            value_module=vm, dim=D, a=a, lr=LR,
            loss_type="quad", analytical_value_fn=anal_fn,
        )
    else:
        ds = OnPolicySMCDataset(
            dim=D, drift=base_drift, value=vm, smc_value=smc_reward,
            reward=reward, device=DEVICE, a=a,
            batch_size=cfg["ds_batch"], n_steps=100,
            mc_samples_per_step=cfg["mc"],
            sampling_method="ancestral_td_lambda", lambda_eff=0.0,
        )
        loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
        model = OnPolicyValue(
            base_score_module=base_drift, value_module=vm,
            reward_function=reward, dim=D, a=a, lr=LR,
            loss_type="quad", analytical_value_fn=anal_fn,
        )

    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{run_name}", save_last=True, save_top_k=1,
        monitor="val_reward_mean", mode="max", filename="best",
    )
    logger = CSVLogger(LOG_DIR, name=run_name, version=0)
    trainer = L.Trainer(
        max_steps=MAX_STEPS,
        val_check_interval=max(1, MAX_STEPS // 60),
        callbacks=[ckpt_cb], logger=logger,
        enable_checkpointing=True, enable_progress_bar=True,
    )

    t0 = time.perf_counter()
    trainer.fit(model, loader, val_dataloaders=val_loader)
    elapsed = time.perf_counter() - t0
    print(f"  Elapsed: {elapsed/60:.1f} min")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\n\n{'='*70}")
print(f"  RESULTS (E_OPT = {E_OPT:.4f})")
print(f"{'='*70}")

print(f"\n  {'Run':<15} {'mc':>4} {'Best Rwd':>10} {'Final Rwd':>11} {'Gap':>8}")
print(f"  {'-'*52}")

run_data = {}
for cfg in RUNS:
    rn = cfg["name"]
    csv = f"{LOG_DIR}/{rn}/version_0/metrics.csv"
    if not os.path.exists(csv): continue
    df = pd.read_csv(csv)
    val = df.dropna(subset=["val_reward_mean"])
    if len(val) == 0: continue
    best = val["val_reward_mean"].max()
    final = val["val_reward_mean"].iloc[-1]
    gap = E_OPT - best
    mc_str = str(cfg["mc"]) if cfg["mc"] else "—"
    print(f"  {rn:<15} {mc_str:>4} {best:>10.4f} {final:>11.4f} {gap:>8.4f}")
    run_data[rn] = (val["step"].values, val["val_reward_mean"].values)

# ---------------------------------------------------------------------------
# Plot 1: Terminal reward vs steps
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_title("mc_samples Sweep: Terminal Reward vs Training Steps\n(ancestral_td_lambda λ=0, smc=reward)",
             fontsize=13, fontweight="bold")

cmap = plt.cm.viridis
for i, cfg in enumerate(RUNS):
    rn = cfg["name"]
    if rn not in run_data: continue
    steps, rewards = run_data[rn]
    if cfg["mc"] is None:
        ax.plot(steps, rewards, color="black", linestyle="--", linewidth=2.5, label="off-policy")
    else:
        color = cmap(i / len(RUNS))
        ax.plot(steps, rewards, color=color, linewidth=1.5, label=f"mc={cfg['mc']}")

ax.axhline(E_OPT, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label=f"E_opt={E_OPT:.3f}")
ax.set_xlabel("Training Steps")
ax.set_ylabel("Avg Terminal Reward")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("notebooks/mc_samples_sweep_reward.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: notebooks/mc_samples_sweep_reward.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 2: Best reward vs mc_samples
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title("Best Terminal Reward vs mc_samples", fontsize=13, fontweight="bold")

mc_vals = []
best_vals = []
for cfg in RUNS:
    if cfg["mc"] is None: continue
    rn = cfg["name"]
    if rn not in run_data: continue
    mc_vals.append(cfg["mc"])
    best_vals.append(run_data[rn][1].max())

ax.plot(mc_vals, best_vals, "o-", color="blue", linewidth=2, markersize=8)
if "offpolicy" in run_data:
    ax.axhline(run_data["offpolicy"][1].max(), color="black", linestyle="--",
               linewidth=1.5, label="off-policy best")
ax.axhline(E_OPT, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label=f"E_opt={E_OPT:.3f}")
ax.set_xlabel("mc_samples")
ax.set_ylabel("Best Terminal Reward")
ax.set_xscale("log")
ax.set_xticks(mc_vals)
ax.set_xticklabels([str(m) for m in mc_vals])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("notebooks/mc_samples_sweep_best.png", dpi=150, bbox_inches="tight")
print("Saved: notebooks/mc_samples_sweep_best.png")
plt.close()

print(f"\nDone. E_OPT = {E_OPT:.4f}")
