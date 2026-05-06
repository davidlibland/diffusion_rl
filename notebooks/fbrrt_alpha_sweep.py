#!/usr/bin/env python3
"""FBRRT alpha sweep: 0 (off-policy, base drift), 0.5 (halfway), 1 (on-policy).

alpha=0 particles follow the base SDE drift f (no policy correction), so V is
learned on-policy w.r.t. its own definition V(x,t) = log E[exp(r(X_T)) | base].
This avoids Sutton's deadly triad (function approximation + bootstrapping
+ off-policy divergence).

alpha=1 is standard FBRRT (particles guided by f + 2a·∇V).
alpha=0.5 is the midpoint (no driver correction).

Plus an off-policy baseline for comparison.
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

with open("notebooks/analytical_target.json") as f: _at = json.load(f)
E_OPT = _at["E_opt"]
all_rewards = reward_fn(torch.from_numpy(X).float()); max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()

# ── Config ─────────────────────────────────────────────────────────────────
LOG_DIR = "lightning_logs/fbrrt_alpha"
CKPT_DIR = "checkpoints/fbrrt_alpha"
TOTAL_STEPS = 3000
WS_STEPS = 1000  # Off-policy warm-start before FBRRT (stabilizes ∇V)
BS = 256
LR = 1e-3
EMA_DECAY = 0.999
GRAD_CLIP = 1.0
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

# ── Configs ──────────────────────────────────────────────────────────────
configs = [
    {"name": "offpolicy",       "method": "offpolicy"},
    {"name": "fbrrt_alpha0",    "method": "fbrrt", "alpha": 0.0},
    {"name": "fbrrt_alpha0.5",  "method": "fbrrt", "alpha": 0.5},
    {"name": "fbrrt_alpha1",    "method": "fbrrt", "alpha": 1.0},
]

print(f"E_OPT={E_OPT:.4f}")
print(f"Total configs: {len(configs)}")

# ── Run ──────────────────────────────────────────────────────────────────
for i, cfg in enumerate(configs):
    name = cfg["name"]
    csv_path = f"{LOG_DIR}/{name}/version_0/metrics.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        val = df.dropna(subset=["val_reward_mean"])
        if len(val) > 0 and int(val["step"].max()) >= TOTAL_STEPS - 1:
            print(f"  {name}: already complete, skipping.")
            continue

    for v in range(3):
        p = f"{LOG_DIR}/{name}/version_{v}"
        if os.path.exists(p): shutil.rmtree(p)
    p = f"{CKPT_DIR}/{name}"
    if os.path.exists(p): shutil.rmtree(p)

    print(f"\n{'='*70}\n  [{i+1}/{len(configs)}] {name}\n{'='*70}")
    t0 = time.time()
    vm = ValueNetwork(D, bias=bias_val)

    if cfg["method"] == "offpolicy":
        ds_off = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=1024)
        off_model = OffPolicyValue(
            base_score_module=base_drift, value_module=vm,
            reward_function=reward_fn, dim=D, a=a, lr=LR,
            loss_type="quad", analytical_value_fn=anal_fn,
        )
        loader = DataLoader(ds_off, batch_size=BS)
        tcb = TrajCB(anal_fn)
        ccb = ModelCheckpoint(dirpath=f"{CKPT_DIR}/{name}", save_last=True, save_top_k=1,
                              monitor="val_reward_mean", mode="max", filename="best")
        logger = CSVLogger(LOG_DIR, name=name, version=0)
        trainer = L.Trainer(max_steps=TOTAL_STEPS, val_check_interval=max(1, TOTAL_STEPS // 40),
                            callbacks=[ccb, tcb], logger=logger, gradient_clip_val=GRAD_CLIP,
                            enable_checkpointing=True, enable_progress_bar=True)
        trainer.fit(off_model, loader, val_dataloaders=val_loader)
        del off_model
    else:
        # Phase 1: off-policy warm-start to stabilize ∇V
        model = OnPolicyValue(
            base_score_module=base_drift, value_module=vm,
            reward_function=reward_fn, dim=D, a=a, lr=LR,
            loss_type="quad", analytical_value_fn=anal_fn, ema_decay=EMA_DECAY,
        )
        print(f"  Phase 1: off-policy warm-start for {WS_STEPS} steps...")
        ds_off = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=1024)
        off_warm = OffPolicyValue(
            base_score_module=base_drift, value_module=vm,
            reward_function=reward_fn, dim=D, a=a, lr=LR,
            loss_type="quad", analytical_value_fn=anal_fn,
        )
        ws_loader = DataLoader(ds_off, batch_size=BS)
        ws_tcb = TrajCB(anal_fn)
        ws_logger = CSVLogger(LOG_DIR, name=name, version=0)
        ws_trainer = L.Trainer(
            max_steps=WS_STEPS,
            val_check_interval=max(1, WS_STEPS // 20),
            callbacks=[ws_tcb],
            logger=ws_logger,
            gradient_clip_val=GRAD_CLIP,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )
        ws_trainer.fit(off_warm, ws_loader, val_dataloaders=val_loader)
        # Sync EMA shadow with warm-started value to avoid stale init
        for p_ema, p_live in zip(model.ema.shadow.parameters(), vm.parameters()):
            p_ema.data.copy_(p_live.data)
        del off_warm, ws_loader, ws_trainer
        gc.collect()
        print(f"  Phase 2: FBRRT α={cfg['alpha']} for {TOTAL_STEPS - WS_STEPS} steps...")
        ds = OnPolicySMCDataset(
            dim=D, drift=base_drift, value=model.ema,
            smc_value=smc_reward, reward=reward_fn,
            device=DEVICE, a=a,
            batch_size=4, n_steps=100, mc_samples_per_step=10,
            sampling_method="fbrrt",
            lambda_eff=0.0,
            branch=8, entropy_lambda=2.0,
            fbrrt_alpha=cfg["alpha"],
            off_policy_frac=0.0,
            generating_function=gmm_sample,
        )
        loader = DataLoader(ds, batch_size=BS)
        tcb = TrajCB(anal_fn)
        ccb = ModelCheckpoint(dirpath=f"{CKPT_DIR}/{name}", save_last=True, save_top_k=1,
                              monitor="val_reward_mean", mode="max", filename="best")
        # Log FBRRT phase as version 1 (warm-start is version 0)
        logger = CSVLogger(LOG_DIR, name=name, version=1)
        fb_steps = TOTAL_STEPS - WS_STEPS
        trainer = L.Trainer(max_steps=fb_steps, val_check_interval=max(1, fb_steps // 40),
                            callbacks=[ccb, tcb], logger=logger, gradient_clip_val=GRAD_CLIP,
                            enable_checkpointing=True, enable_progress_bar=True)
        trainer.fit(model, loader, val_dataloaders=val_loader)
        del model

    elapsed = (time.time() - t0) / 60
    print(f"  Elapsed: {elapsed:.1f} min")
    del vm, trainer, loader
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ── Results ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}\n  RESULTS (E_OPT={E_OPT:.4f})\n{'='*70}\n")
all_data = {}
print(f"  {'Name':<20} {'Best':>8} {'Final':>8} {'G-MAE':>7} {'B-MAE':>7} {'Stable':>7}")
print(f"  {'-'*60}")

for cfg in configs:
    name = cfg["name"]
    dfs = []
    for v in [0, 1]:
        csv_path = f"{LOG_DIR}/{name}/version_{v}/metrics.csv"
        if os.path.exists(csv_path):
            d = pd.read_csv(csv_path)
            if v == 1:
                d = d.copy()
                d["step"] = d["step"] + WS_STEPS
            dfs.append(d)
    if not dfs: continue
    df = pd.concat(dfs, ignore_index=True)
    val = df.dropna(subset=["val_reward_mean"])
    if len(val) == 0: continue
    best = val["val_reward_mean"].max()
    final = val["val_reward_mean"].iloc[-1]
    gm = df.dropna(subset=["traj_avg_mae_guided"])
    bm = df.dropna(subset=["traj_avg_mae_base"])
    gf = gm["traj_avg_mae_guided"].iloc[-1] if len(gm) > 0 else float("nan")
    bf = bm["traj_avg_mae_base"].iloc[-1] if len(bm) > 0 else float("nan")
    stable = "YES" if abs(final - best) < 5 else "no"
    all_data[name] = {"cfg": cfg, "best": best, "final": final, "g_mae": gf, "b_mae": bf, "stable": stable}
    print(f"  {name:<20} {best:>8.3f} {final:>8.3f} {gf:>7.3f} {bf:>7.3f} {stable:>7}")

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("FBRRT alpha sweep: off-policy (α=0) vs on-policy (α=1)", fontsize=13, fontweight="bold")

styles = {
    "offpolicy":      dict(color="black", ls="--", lw=2.5, label="off-policy"),
    "fbrrt_alpha0":   dict(color="#2a9d8f", ls="-", lw=2,   label="FBRRT α=0 (base drift)"),
    "fbrrt_alpha0.5": dict(color="#f4a261", ls="-", lw=2,   label="FBRRT α=0.5"),
    "fbrrt_alpha1":   dict(color="#e63946", ls="-", lw=2,   label="FBRRT α=1 (on-policy)"),
}

for ax_i, (metric, title, ylabel) in enumerate([
    ("val_reward_mean", "Terminal Reward", "Avg Terminal Reward"),
    ("traj_avg_mae_guided", "Guided Trajectory MAE", "Avg |V_model - V_analytical|"),
]):
    ax = axes[ax_i]
    ax.set_title(title)
    for cfg in configs:
        name = cfg["name"]
        dfs = []
        for v in [0, 1]:
            csv_path = f"{LOG_DIR}/{name}/version_{v}/metrics.csv"
            if os.path.exists(csv_path):
                d = pd.read_csv(csv_path)
                if v == 1:
                    d = d.copy()
                    d["step"] = d["step"] + WS_STEPS
                dfs.append(d)
        if not dfs: continue
        df = pd.concat(dfs, ignore_index=True)
        sub = df.dropna(subset=[metric])
        if len(sub) == 0: continue
        s = styles.get(name, dict(color="gray", ls="-", lw=1))
        ax.plot(sub["step"], sub[metric], **s)
        # Mark warm-start transition for FBRRT runs
        if name != "offpolicy" and ax_i == 0:
            ax.axvline(WS_STEPS, color="gray", ls=":", alpha=0.3)
    ax.set_xlabel("Training Steps"); ax.set_ylabel(ylabel)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    if metric == "val_reward_mean":
        ax.axhline(E_OPT, color="red", ls=":", alpha=0.3, label=f"E_opt={E_OPT:.2f}")
    else:
        ax.set_yscale("log")

plt.tight_layout()
plt.savefig("notebooks/fbrrt_alpha_sweep.png", dpi=150, bbox_inches="tight")
print("\nSaved: notebooks/fbrrt_alpha_sweep.png")

with open("notebooks/fbrrt_alpha_sweep_results.json", "w") as f:
    json.dump({"results": {k: {kk: vv for kk, vv in v.items() if kk != "cfg"}
                           for k, v in all_data.items()}}, f, indent=2, default=str)

print("\nDone.")
