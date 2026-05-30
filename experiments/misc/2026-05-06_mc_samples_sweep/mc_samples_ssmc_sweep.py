#!/usr/bin/env python3
"""Single-seed MC sweep: off_policy_frac × mc_samples."""

import gc, json, os, shutil, time
from functools import partial
from math import sqrt, ceil

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

# ── GMM setup ──────────────────────────────────────────────────────────────
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler(); X = scalar.fit_transform(X)
clf = GaussianMixture(n_components=100, covariance_type="spherical"); clf.fit(X)
_means = torch.from_numpy(clf.means_).double()
_sigma2 = torch.from_numpy(clf.covariances_).double()
_weights = torch.from_numpy(clf.weights_).double()
_sigmas = torch.sqrt(_sigma2)[:, None]
_weights_col = _weights[:, None]
D = 2; a = 1.0; c = torch.tensor([1.0, 0.0])
means_np = clf.means_; sigmas_np = np.sqrt(clf.covariances_); weights_np = clf.weights_

# ── Analytical value ───────────────────────────────────────────────────────
class AV(nn.Module):
    def __init__(self, means, sigma2, weights, a=1.0, c_=None, D_=2):
        super().__init__()
        if c_ is None: c_ = torch.tensor([1.0, 0.0])
        self.register_buffer("means", means.float())
        self.register_buffer("sigma2", sigma2.float())
        self.register_buffer("weights", weights.float())
        self.a = a; self.D = D_; self.register_buffer("c", c_.float())
    def _log_Z(self, m, v):
        cc = self.c.double(); d = 1+20*v
        return (-self.D/2*torch.log(d)+(-10*(m**2).sum(-1)+20*(m*cc).sum(-1)+200*v*(cc**2).sum())/d-10*(cc**2).sum())
    def forward(self, x, t):
        x = x.double(); t = t.double().reshape(-1)
        if t.numel() == 1: t = t.expand(x.shape[0])
        t_ = t[:, None]; m = self.means.double(); s2 = self.sigma2.double(); w = self.weights.double()
        dk = t_*s2[None,:]+2*self.a*(1-t_); mm = t_[:,:,None]*m[None,:,:]
        d2 = ((x[:,None,:]-mm)**2).sum(-1); ts = t_+1e-40
        lg = (-self.D/2*torch.log(2*torch.pi*ts*dk)-d2/(2*ts*dk))
        lw = torch.log(w)[None,:]; lpw = lw+lg-torch.logsumexp(lw+lg,dim=1,keepdim=True)
        tV = 2*self.a*(1-t_)*s2[None,:]/dk
        tmu = (s2[None,:,None]*x[:,None,:]+2*self.a*(1-t_)[:,:,None]*m[None,:,:])/dk[:,:,None]
        return torch.logsumexp(lpw+self._log_Z(tmu, tV), dim=1).float()

_avm = AV(_means, _sigma2, _weights, a=a, c_=c, D_=D)
def anal_fn(x, t): return _avm(x.cpu(), t.cpu()).to(x.device)

# ── Drift ──────────────────────────────────────────────────────────────────
def gmm_drift(xt, ts, a_):
    ts = ts.reshape(-1, 1)
    xt_ = xt[..., None]; means_ = _means.float().to(xt).T[None, ...]
    ts_ = ts[..., None]; sigmas_ = _sigmas.float().to(xt).T; weights_ = _weights_col.float().to(xt).T
    denom = 2*a_*(1-ts)+ts*sigmas_**2
    le = -reduce((xt_-means_*ts_)**2, "n d m -> n m", "sum")/(2*ts*denom)
    lsf = torch.log(2*a_*(1-ts)/denom)*D/2
    lrw = torch.log(weights_)+le+lsf; lw = lrw-torch.logsumexp(lrw, dim=1, keepdim=True)
    lw = torch.where((ts == 0), torch.log(weights_), lw)
    nm = (2*a_*(1-ts_)*means_+xt_*sigmas_[None,...]**2)/denom[:,None,:]
    us = (nm-xt[:,:,None])/(1-ts[...,None])
    return reduce(torch.exp(lw)[:,None,:]*us, "n d m -> n d", "sum")

DEVICE = "mps"
base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)
reward_fn = lambda x: -10*(x - c.to(x)).square().sum(dim=1)
smc_reward = lambda x, t: reward_fn(x)
def gmm_sample(n):
    k = np.random.choice(len(weights_np), size=n, p=weights_np)
    return means_np[k] + sigmas_np[k, np.newaxis]*np.random.randn(n, D)

with open("experiments/common/analytical_target.json") as f: _at = json.load(f)
E_OPT = _at["E_opt"]
all_rewards = reward_fn(torch.from_numpy(X).float()); max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()

# ── Config ─────────────────────────────────────────────────────────────────
LOG_DIR = "lightning_logs/ssmc_sweep"
CKPT_DIR = "checkpoints/ssmc_sweep"
TOTAL_STEPS = 3000
BS = 256
LR = 1e-3
EMA_DECAY = 0.999
val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

# ── Trajectory eval callback ──────────────────────────────────────────────
class TrajCB(Callback):
    def __init__(self, af, n=256, ns=100):
        super().__init__(); self.af = af; self.n = n; self.ns = ns
    def on_validation_batch_end(self, trainer, pl, outputs, batch, bi):
        if bi > 0: return
        dim = pl.hparams.dim; dev = pl.device; n = self.n; dt = 1.0/self.ns
        for beta, label in [(0, "base"), (1, "guided")]:
            x = torch.zeros(n, dim, device=dev)
            ax, at = [x], [torch.zeros(n, device=dev)]
            dfn = partial(pl.drift, beta=beta)
            for st in torch.linspace(0, 1, self.ns+1, device=dev)[:-1]:
                tv = st.expand(n)
                dx = dfn(x, tv)*dt; db = sqrt(2*pl.a*dt)*torch.randn_like(x)
                x = x + dx + db; ax.append(x)
                at.append(torch.full((n,), float(st)+dt, device=dev))
            ax = torch.cat(ax); at = torch.cat(at)
            with torch.no_grad():
                vp = pl.value_module(ax, at); va = self.af(ax, at)
            err = vp - va
            pl.log(f"traj_avg_mae_{label}", err.abs().mean(), prog_bar=False)

# ── Sweep configs ──────────────────────────────────────────────────────────
configs = []

# 1) Pure off-policy baseline
configs.append({"name": "offpolicy", "method": "offpolicy", "mc": 10, "frac": 1.0, "lr": LR})

# 2) Sweep off_policy_frac × mc_samples
for mc in [2, 10, 30]:
    for frac in [0.0, 0.25, 0.5, 0.75]:
        configs.append({
            "name": f"ssmc_mc{mc}_frac{int(frac*100)}",
            "method": "single_seed_mc",
            "mc": mc,
            "frac": frac,
            "lr": LR,
        })

# 3) A few LR variants at mc=10, frac=0.5
for lr in [3e-4, 3e-3]:
    configs.append({
        "name": f"ssmc_mc10_frac50_lr{lr:.0e}",
        "method": "single_seed_mc",
        "mc": 10,
        "frac": 0.5,
        "lr": lr,
    })

print(f"E_OPT={E_OPT:.4f}")
print(f"Total configs: {len(configs)}")
for cfg in configs:
    print(f"  {cfg['name']}")

# ── Run each config ───────────────────────────────────────────────────────
for i, cfg in enumerate(configs):
    name = cfg["name"]
    csv_path = f"{LOG_DIR}/{name}/version_0/metrics.csv"

    # Skip if already done
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        val = df.dropna(subset=["val_reward_mean"])
        if len(val) > 0 and int(val["step"].max()) >= TOTAL_STEPS - 1:
            print(f"\n  {name}: Already complete, skipping.")
            continue

    # Clean up partial
    for v in range(3):
        p = f"{LOG_DIR}/{name}/version_{v}"
        if os.path.exists(p): shutil.rmtree(p)
    p = f"{CKPT_DIR}/{name}"
    if os.path.exists(p): shutil.rmtree(p)

    print(f"\n{'='*70}")
    print(f"  [{i+1}/{len(configs)}] {name}")
    print(f"  method={cfg['method']} mc={cfg['mc']} frac={cfg['frac']} lr={cfg['lr']}")
    print(f"{'='*70}")

    t0 = time.time()
    vm = ValueNetwork(D, bias=bias_val)

    if cfg["method"] == "offpolicy":
        ds_off = InterpolatingNumpyDataset(
            generating_function=gmm_sample, a=a, batch_size=1024
        )
        off_model = OffPolicyValue(
            base_score_module=base_drift, value_module=vm,
            reward_function=reward_fn, dim=D, a=a, lr=cfg["lr"],
            loss_type="quad", analytical_value_fn=anal_fn,
        )
        loader = DataLoader(ds_off, batch_size=BS)
        tcb = TrajCB(anal_fn)
        ccb = ModelCheckpoint(dirpath=f"{CKPT_DIR}/{name}", save_last=True,
                              save_top_k=1, monitor="val_reward_mean", mode="max", filename="best")
        logger = CSVLogger(LOG_DIR, name=name, version=0)
        trainer = L.Trainer(
            max_steps=TOTAL_STEPS,
            val_check_interval=max(1, TOTAL_STEPS // 60),
            callbacks=[ccb, tcb], logger=logger,
            enable_checkpointing=True, enable_progress_bar=True,
        )
        trainer.fit(off_model, loader, val_dataloaders=val_loader)
        del off_model
    else:
        model = OnPolicyValue(
            base_score_module=base_drift, value_module=vm,
            reward_function=reward_fn, dim=D, a=a, lr=cfg["lr"],
            loss_type="quad", analytical_value_fn=anal_fn, ema_decay=EMA_DECAY,
        )
        ds_batch = max(1, ceil(32 / cfg["mc"]))
        ds = OnPolicySMCDataset(
            dim=D, drift=base_drift, value=model.ema,
            smc_value=smc_reward, reward=reward_fn,
            device=DEVICE, a=a,
            batch_size=ds_batch, n_steps=100,
            mc_samples_per_step=cfg["mc"],
            sampling_method="single_seed_mc",
            off_policy_frac=cfg["frac"],
            generating_function=gmm_sample,
        )
        loader = DataLoader(ds, batch_size=BS)
        tcb = TrajCB(anal_fn)
        ccb = ModelCheckpoint(dirpath=f"{CKPT_DIR}/{name}", save_last=True,
                              save_top_k=1, monitor="val_reward_mean", mode="max", filename="best")
        logger = CSVLogger(LOG_DIR, name=name, version=0)
        trainer = L.Trainer(
            max_steps=TOTAL_STEPS,
            val_check_interval=max(1, TOTAL_STEPS // 60),
            callbacks=[ccb, tcb], logger=logger,
            enable_checkpointing=True, enable_progress_bar=True,
        )
        trainer.fit(model, loader, val_dataloaders=val_loader)
        del model

    elapsed = (time.time() - t0) / 60
    print(f"  Elapsed: {elapsed:.1f} min")

    # Cleanup
    del vm, trainer, loader
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ── Gather results ────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  RESULTS ({TOTAL_STEPS} steps, E_OPT={E_OPT:.4f})")
print(f"{'='*70}\n")

all_data = {}
print(f"  {'Name':<30} {'Best':>8} {'Final':>8} {'G-MAE':>7} {'B-MAE':>7}")
print(f"  {'-'*64}")

for cfg in configs:
    name = cfg["name"]
    csv_path = f"{LOG_DIR}/{name}/version_0/metrics.csv"
    if not os.path.exists(csv_path): continue
    df = pd.read_csv(csv_path)
    val = df.dropna(subset=["val_reward_mean"])
    if len(val) == 0: continue
    best = val["val_reward_mean"].max()
    final = val["val_reward_mean"].iloc[-1]
    gm = df.dropna(subset=["traj_avg_mae_guided"])
    bm = df.dropna(subset=["traj_avg_mae_base"])
    gf = gm["traj_avg_mae_guided"].iloc[-1] if len(gm) > 0 else float("nan")
    bf = bm["traj_avg_mae_base"].iloc[-1] if len(bm) > 0 else float("nan")
    all_data[name] = {"cfg": cfg, "best": best, "final": final, "g_mae": gf, "b_mae": bf}
    print(f"  {name:<30} {best:>8.3f} {final:>8.3f} {gf:>7.3f} {bf:>7.3f}")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Single-Seed MC: Off-Policy Fraction × MC Samples", fontsize=14, fontweight="bold")

colors_frac = {0.0: "#e63946", 0.25: "#f4a261", 0.5: "#2a9d8f", 0.75: "#264653", 1.0: "black"}
ls_mc = {2: ":", 10: "-", 30: "--"}

# Terminal reward
ax = axes[0]
ax.set_title("Terminal Reward vs Steps")
csv_off = f"{LOG_DIR}/offpolicy/version_0/metrics.csv"
if os.path.exists(csv_off):
    df = pd.read_csv(csv_off)
    val = df.dropna(subset=["val_reward_mean"])
    ax.plot(val["step"], val["val_reward_mean"], color="black", lw=2.5, ls="--", label="off-policy")
for cfg in configs:
    if cfg["method"] != "single_seed_mc": continue
    if "lr" in cfg["name"] and "frac50" in cfg["name"]: continue
    name = cfg["name"]
    csv = f"{LOG_DIR}/{name}/version_0/metrics.csv"
    if not os.path.exists(csv): continue
    df = pd.read_csv(csv)
    val = df.dropna(subset=["val_reward_mean"])
    col = colors_frac.get(cfg["frac"], "gray")
    ls = ls_mc.get(cfg["mc"], "-")
    ax.plot(val["step"], val["val_reward_mean"], color=col, ls=ls, lw=1.5,
            label=f"mc={cfg['mc']} frac={cfg['frac']}")
ax.axhline(E_OPT, color="red", ls=":", alpha=0.3, label=f"E_opt={E_OPT:.2f}")
ax.set_xlabel("Steps"); ax.set_ylabel("Avg Terminal Reward")
ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

# Guided MAE
ax = axes[1]
ax.set_title("V Error on Guided Trajectories")
if os.path.exists(csv_off):
    df = pd.read_csv(csv_off)
    sub = df.dropna(subset=["traj_avg_mae_guided"])
    ax.plot(sub["step"], sub["traj_avg_mae_guided"], color="black", lw=2.5, ls="--", label="off-policy")
for cfg in configs:
    if cfg["method"] != "single_seed_mc": continue
    if "lr" in cfg["name"] and "frac50" in cfg["name"]: continue
    name = cfg["name"]
    csv = f"{LOG_DIR}/{name}/version_0/metrics.csv"
    if not os.path.exists(csv): continue
    df = pd.read_csv(csv)
    sub = df.dropna(subset=["traj_avg_mae_guided"])
    if len(sub) == 0: continue
    col = colors_frac.get(cfg["frac"], "gray")
    ls = ls_mc.get(cfg["mc"], "-")
    ax.plot(sub["step"], sub["traj_avg_mae_guided"], color=col, ls=ls, lw=1.5,
            label=f"mc={cfg['mc']} frac={cfg['frac']}")
ax.set_xlabel("Steps"); ax.set_ylabel("Avg MAE"); ax.set_yscale("log")
ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("experiments/misc/2026-05-06_mc_samples_sweep/mc_samples_sweep_reward.png", dpi=150, bbox_inches="tight")
print("\nSaved: experiments/misc/2026-05-06_mc_samples_sweep/mc_samples_sweep_reward.png")

# ── Summary heatmap: best reward by frac × mc ────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

for ax_i, (metric, title) in enumerate([("best", "Best Terminal Reward"), ("g_mae", "Final Guided MAE")]):
    ax = axes2[ax_i]
    fracs = [0.0, 0.25, 0.5, 0.75]
    mcs = [2, 10, 30]
    grid = np.full((len(fracs), len(mcs)), np.nan)
    for fi, frac in enumerate(fracs):
        for mi, mc in enumerate(mcs):
            name = f"ssmc_mc{mc}_frac{int(frac*100)}"
            if name in all_data:
                grid[fi, mi] = all_data[name][metric]
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn" if metric == "best" else "RdYlGn_r")
    ax.set_xticks(range(len(mcs))); ax.set_xticklabels(mcs)
    ax.set_yticks(range(len(fracs))); ax.set_yticklabels([f"{f:.0%}" for f in fracs])
    ax.set_xlabel("mc_samples"); ax.set_ylabel("off_policy_frac")
    ax.set_title(title)
    for fi in range(len(fracs)):
        for mi in range(len(mcs)):
            if not np.isnan(grid[fi, mi]):
                ax.text(mi, fi, f"{grid[fi, mi]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("experiments/misc/2026-05-06_mc_samples_sweep/mc_samples_sweep_best.png", dpi=150, bbox_inches="tight")
print("Saved: experiments/misc/2026-05-06_mc_samples_sweep/mc_samples_sweep_best.png")

# Save JSON
with open("experiments/misc/2026-05-06_mc_samples_sweep/mc_samples_sweep_results.json", "w") as f:
    json.dump({"configs": configs, "results": {k: {kk: vv for kk, vv in v.items() if kk != "cfg"} for k, v in all_data.items()}}, f, indent=2, default=str)

print("\nDone.")
