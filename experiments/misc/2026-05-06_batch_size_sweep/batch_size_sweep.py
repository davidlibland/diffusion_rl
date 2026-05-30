#!/usr/bin/env python3
"""Batch size sweep: does smaller training batch size help on-policy methods?

Hypothesis: SSMC/FBRRT have lower-variance targets, but large batch size (256)
averages over many samples per gradient step, negating this advantage. Smaller
batch sizes may let the lower variance shine through.

Training batch sizes: 1, 2, 4, 16, 256 (squares: sqrt(var) scales as sqrt(BS)).
Dataset internal generation batch size is unchanged.

Sweeps training batch size × off_policy_frac for SSMC (mc=30), FBRRT (branch=8),
and ancestral MC TD(λ=0).
"""

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
LOG_DIR = "lightning_logs/batch_sweep"
CKPT_DIR = "checkpoints/batch_sweep"
TOTAL_STEPS = 3000
LR = 1e-3
EMA_DECAY = 0.999
val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

# Training batch sizes: 1, 2, 4, 16, 256 (squares progression)
BATCH_SIZES = [1, 2, 4, 16, 256]

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

# ── Build configs ──────────────────────────────────────────────────────────
configs = []

# Off-policy baselines at each batch size
for bs in BATCH_SIZES:
    configs.append({
        "name": f"offpolicy_bs{bs}",
        "method": "offpolicy",
        "sampling_method": None,
        "mc": 10, "branch": 4,
        "frac": 1.0,
        "bs": bs,
    })

# SSMC with mc=30: sweep batch size × frac
for bs in BATCH_SIZES:
    for frac in [0.0, 0.5]:
        configs.append({
            "name": f"ssmc_bs{bs}_frac{int(frac*100)}",
            "method": "on_policy",
            "sampling_method": "single_seed_mc",
            "mc": 30, "branch": 4,
            "frac": frac,
            "bs": bs,
        })

# FBRRT with branch=8: sweep batch size × frac
for bs in BATCH_SIZES:
    for frac in [0.0, 0.5]:
        configs.append({
            "name": f"fbrrt_bs{bs}_frac{int(frac*100)}",
            "method": "on_policy",
            "sampling_method": "fbrrt",
            "mc": 10, "branch": 8,
            "frac": frac,
            "bs": bs,
        })

# Ancestral MC TD(λ=0): sweep batch size × frac
for bs in BATCH_SIZES:
    for frac in [0.0, 0.5]:
        configs.append({
            "name": f"amctd_bs{bs}_frac{int(frac*100)}",
            "method": "on_policy",
            "sampling_method": "ancestral_mc_td_lambda",
            "mc": 10, "branch": 4,
            "frac": frac,
            "bs": bs,
        })

print(f"E_OPT={E_OPT:.4f}")
print(f"Total configs: {len(configs)}")

# ── Run ───────────────────────────────────────────────────────────────────
for i, cfg in enumerate(configs):
    name = cfg["name"]
    csv_path = f"{LOG_DIR}/{name}/version_0/metrics.csv"

    # Skip if done
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        val = df.dropna(subset=["val_reward_mean"])
        if len(val) > 0 and int(val["step"].max()) >= TOTAL_STEPS - 1:
            print(f"  {name}: Already complete, skipping.")
            continue

    # Clean partial
    for v in range(3):
        p = f"{LOG_DIR}/{name}/version_{v}"
        if os.path.exists(p): shutil.rmtree(p)
    p = f"{CKPT_DIR}/{name}"
    if os.path.exists(p): shutil.rmtree(p)

    print(f"\n{'='*70}")
    print(f"  [{i+1}/{len(configs)}] {name}  (bs={cfg['bs']})")
    print(f"{'='*70}")

    t0 = time.time()
    vm = ValueNetwork(D, bias=bias_val)
    bs = cfg["bs"]

    if cfg["method"] == "offpolicy":
        ds_off = InterpolatingNumpyDataset(
            generating_function=gmm_sample, a=a, batch_size=1024
        )
        off_model = OffPolicyValue(
            base_score_module=base_drift, value_module=vm,
            reward_function=reward_fn, dim=D, a=a, lr=LR,
            loss_type="quad", analytical_value_fn=anal_fn,
        )
        loader = DataLoader(ds_off, batch_size=bs)
        tcb = TrajCB(anal_fn)
        ccb = ModelCheckpoint(dirpath=f"{CKPT_DIR}/{name}", save_last=True,
                              save_top_k=1, monitor="val_reward_mean", mode="max", filename="best")
        logger = CSVLogger(LOG_DIR, name=name, version=0)
        trainer = L.Trainer(
            max_steps=TOTAL_STEPS,
            val_check_interval=max(1, TOTAL_STEPS // 40),
            callbacks=[ccb, tcb], logger=logger,
            enable_checkpointing=True, enable_progress_bar=True,
        )
        trainer.fit(off_model, loader, val_dataloaders=val_loader)
        del off_model
    else:
        model = OnPolicyValue(
            base_score_module=base_drift, value_module=vm,
            reward_function=reward_fn, dim=D, a=a, lr=LR,
            loss_type="quad", analytical_value_fn=anal_fn, ema_decay=EMA_DECAY,
        )
        ds_batch = max(1, ceil(32 / cfg["mc"]))
        ds = OnPolicySMCDataset(
            dim=D, drift=base_drift, value=model.ema,
            smc_value=smc_reward, reward=reward_fn,
            device=DEVICE, a=a,
            batch_size=ds_batch, n_steps=100,
            mc_samples_per_step=cfg["mc"],
            sampling_method=cfg["sampling_method"],
            lambda_eff=0.0,
            branch=cfg["branch"],
            entropy_lambda=2.0,
            off_policy_frac=cfg["frac"],
            generating_function=gmm_sample,
        )
        loader = DataLoader(ds, batch_size=bs)
        tcb = TrajCB(anal_fn)
        ccb = ModelCheckpoint(dirpath=f"{CKPT_DIR}/{name}", save_last=True,
                              save_top_k=1, monitor="val_reward_mean", mode="max", filename="best")
        logger = CSVLogger(LOG_DIR, name=name, version=0)
        trainer = L.Trainer(
            max_steps=TOTAL_STEPS,
            val_check_interval=max(1, TOTAL_STEPS // 40),
            callbacks=[ccb, tcb], logger=logger,
            enable_checkpointing=True, enable_progress_bar=True,
        )
        trainer.fit(model, loader, val_dataloaders=val_loader)
        del model

    elapsed = (time.time() - t0) / 60
    print(f"  Elapsed: {elapsed:.1f} min")
    del vm, trainer, loader
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ── Results ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  RESULTS ({TOTAL_STEPS} steps, E_OPT={E_OPT:.4f})")
print(f"{'='*70}\n")

all_data = {}
print(f"  {'Name':<30} {'BS':>4} {'Best':>8} {'Final':>8} {'G-MAE':>7}")
print(f"  {'-'*60}")

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
    gf = gm["traj_avg_mae_guided"].iloc[-1] if len(gm) > 0 else float("nan")
    all_data[name] = {"cfg": cfg, "best": best, "final": final, "g_mae": gf}
    print(f"  {name:<30} {cfg['bs']:>4} {best:>8.3f} {final:>8.3f} {gf:>7.3f}")

# ── Plots ─────────────────────────────────────────────────────────────────
method_styles = [
    ("offpolicy", "Off-Policy", "black", "--"),
    ("ssmc", "SSMC (mc=30)", "#e63946", "-"),
    ("fbrrt", "FBRRT (br=8)", "#2a9d8f", "-"),
    ("amctd", "AMCTD (λ=0)", "#264653", "-"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
fig.suptitle("Impact of Training Batch Size on On-Policy Methods\n(3000 steps, LR=1e-3, EMA=0.999)",
             fontsize=14, fontweight="bold")

for row, (metric, ylabel) in enumerate([("best", "Best Terminal Reward"), ("g_mae", "Final Guided MAE")]):
    for col, frac in enumerate([0.0, 0.5]):
        ax = axes[row, col]
        frac_label = "pure on-policy" if frac == 0.0 else "50% off-policy"
        ax.set_title(f"{frac_label}")

        for prefix, label, color, ls in method_styles:
            vals = []
            for bs in BATCH_SIZES:
                if prefix == "offpolicy":
                    name = f"offpolicy_bs{bs}"
                else:
                    name = f"{prefix}_bs{bs}_frac{int(frac*100)}"
                if name in all_data:
                    vals.append(all_data[name][metric])
                else:
                    vals.append(float("nan"))
            if prefix == "offpolicy" and col == 1:
                continue  # don't repeat off-policy
            ax.plot(BATCH_SIZES, vals, marker="o", color=color, ls=ls, lw=2, label=label)

        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCH_SIZES)
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_xlabel("Training Batch Size")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        if row == 0:
            ax.axhline(E_OPT, color="red", ls=":", alpha=0.3)

plt.tight_layout()
plt.savefig("experiments/misc/2026-05-06_batch_size_sweep/batch_size_sweep.png", dpi=150, bbox_inches="tight")
print("\nSaved: experiments/misc/2026-05-06_batch_size_sweep/batch_size_sweep.png")

# ── Save JSON ─────────────────────────────────────────────────────────────
with open("experiments/misc/2026-05-06_batch_size_sweep/batch_size_sweep_results.json", "w") as f:
    json.dump({
        "configs": configs,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "cfg"}
                    for k, v in all_data.items()},
    }, f, indent=2, default=str)

print("\nDone.")
