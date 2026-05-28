#!/usr/bin/env python3
"""Geometric k-sweep for single_seed_mc on moons (log_tau = k * h).

Configs (all 1000 steps, BS=256, lr=1e-3, mc=10, n_steps=100):
  - off-policy baseline       (REUSED from lightning_logs/ssmc_vs_offpolicy/offpolicy)
  - single_seed_mc, k = 0.0   (NEW;  log_tau = 0  → uniform SMC weights)
  - single_seed_mc, k = 0.01  (NEW)
  - single_seed_mc, k = 0.1   (NEW)
  - single_seed_mc, k = 1.0   (REUSED from lightning_logs/ssmc_vs_offpolicy/single_seed_mc)

Reused runs are read from prior CSVs and plotted truncated at step ≤ 1000.
"""

import gc
import json
import os
import shutil
import time
from functools import partial
from math import ceil, sqrt

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
from diffusion_rl.modules.resnet_mlp import ValueNetwork


# ── Moons setup (matches ssmc_vs_offpolicy_sweep) ──────────────────────────
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
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


class AV(nn.Module):
    def __init__(self, means, sigma2, weights, a=1.0, c_=None, D_=2):
        super().__init__()
        if c_ is None:
            c_ = torch.tensor([1.0, 0.0])
        self.register_buffer("means", means.float())
        self.register_buffer("sigma2", sigma2.float())
        self.register_buffer("weights", weights.float())
        self.a = a
        self.D = D_
        self.register_buffer("c", c_.float())

    def _log_Z(self, m, v):
        cc = self.c.double()
        d = 1 + 20 * v
        return (
            -self.D / 2 * torch.log(d)
            + (
                -10 * (m**2).sum(-1)
                + 20 * (m * cc).sum(-1)
                + 200 * v * (cc**2).sum()
            )
            / d
            - 10 * (cc**2).sum()
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
        ts = t_ + 1e-40
        lg = -self.D / 2 * torch.log(2 * torch.pi * ts * dk) - d2 / (2 * ts * dk)
        lw = torch.log(w)[None, :]
        lpw = lw + lg - torch.logsumexp(lw + lg, dim=1, keepdim=True)
        tV = 2 * self.a * (1 - t_) * s2[None, :] / dk
        tmu = (
            s2[None, :, None] * x[:, None, :]
            + 2 * self.a * (1 - t_)[:, :, None] * m[None, :, :]
        ) / dk[:, :, None]
        return torch.logsumexp(lpw + self._log_Z(tmu, tV), dim=1).float()


_avm = AV(_means, _sigma2, _weights, a=a, c_=c, D_=D)


def anal_fn(x, t):
    return _avm(x.cpu(), t.cpu()).to(x.device)


def gmm_drift(xt, ts, a_):
    ts = ts.reshape(-1, 1)
    xt_ = xt[..., None]
    means_ = _means.float().to(xt).T[None, ...]
    ts_ = ts[..., None]
    sigmas_ = _sigmas.float().to(xt).T
    weights_ = _weights_col.float().to(xt).T
    denom = 2 * a_ * (1 - ts) + ts * sigmas_**2
    le = -reduce((xt_ - means_ * ts_) ** 2, "n d m -> n m", "sum") / (2 * ts * denom)
    lsf = torch.log(2 * a_ * (1 - ts) / denom) * D / 2
    lrw = torch.log(weights_) + le + lsf
    lw = lrw - torch.logsumexp(lrw, dim=1, keepdim=True)
    lw = torch.where((ts == 0), torch.log(weights_), lw)
    nm = (2 * a_ * (1 - ts_) * means_ + xt_ * sigmas_[None, ...] ** 2) / denom[:, None, :]
    us = (nm - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(torch.exp(lw)[:, None, :] * us, "n d m -> n d", "sum")


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def base_drift(x, t):
    return gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)


def reward_fn(x):
    return -10 * (x - c.to(x)).square().sum(dim=1)


def make_log_tau(k):
    """Return log_tau(x, t) = k * h(x).  k=0 collapses to constant zero."""
    if k == 0.0:
        def _zero(x, t):
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        return _zero

    def _scaled(x, t):
        return k * reward_fn(x)
    return _scaled


def gmm_sample(n):
    k = np.random.choice(len(weights_np), size=n, p=weights_np)
    return means_np[k] + sigmas_np[k, np.newaxis] * np.random.randn(n, D)


with open("notebooks/analytical_target.json") as f:
    _at = json.load(f)
E_OPT = _at["E_opt"]

all_rewards = reward_fn(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (
    torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r
).item()


# ── Config ─────────────────────────────────────────────────────────────────
LOG_DIR = "lightning_logs/ssmc_k_sweep"
CKPT_DIR = "checkpoints/ssmc_k_sweep"

# Reused (from earlier 3000-step run; we truncate to step ≤ 1000 for the plot).
PRIOR_OFF_CSV = "lightning_logs/ssmc_vs_offpolicy/offpolicy/version_0/metrics.csv"
PRIOR_K1_CSV = "lightning_logs/ssmc_vs_offpolicy/single_seed_mc/version_0/metrics.csv"

TOTAL_STEPS = 1000
BS = 256
LR = 1e-3
EMA_DECAY = 0.999
N_STEPS = 100
MC_SAMPLES = 10
OFF_POLICY_FRAC = 0.0
DS_BATCH_ON = max(1, ceil(32 / MC_SAMPLES))

K_GRID_NEW = [0.0, 0.01, 0.1]   # new runs;  k=1 reused from prior

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


# ── Trajectory MAE callback ────────────────────────────────────────────────
class TrajCB(Callback):
    def __init__(self, af, n=256, ns=N_STEPS):
        super().__init__()
        self.af = af
        self.n = n
        self.ns = ns

    def on_validation_batch_end(self, trainer, pl, outputs, batch, bi):
        if bi > 0:
            return
        dim = pl.hparams.dim
        dev = pl.device
        n = self.n
        dt = 1.0 / self.ns
        for beta, label in [(0, "base"), (1, "guided")]:
            x = torch.zeros(n, dim, device=dev)
            ax, at = [x], [torch.zeros(n, device=dev)]
            dfn = partial(pl.drift, beta=beta)
            for st in torch.linspace(0, 1, self.ns + 1, device=dev)[:-1]:
                tv = st.expand(n)
                dx = dfn(x, tv) * dt
                db = sqrt(2 * pl.a * dt) * torch.randn_like(x)
                x = x + dx + db
                ax.append(x)
                at.append(torch.full((n,), float(st) + dt, device=dev))
            ax = torch.cat(ax)
            at = torch.cat(at)
            with torch.no_grad():
                vp = pl.value_module(ax, at)
                va = self.af(ax, at)
            err = vp - va
            pl.log(f"traj_avg_mae_{label}", err.abs().mean(), prog_bar=False)


# ── Run new SSMC configs ───────────────────────────────────────────────────
print(f"E_OPT = {E_OPT:.4f}")
print(f"Device: {DEVICE}")
print(f"BS={BS}  LR={LR}  TOTAL_STEPS={TOTAL_STEPS}  N_STEPS={N_STEPS}")
print(f"single_seed_mc: mc={MC_SAMPLES}  off_policy_frac={OFF_POLICY_FRAC}")
print(f"k_new = {K_GRID_NEW};  k=1 reused from prior run")
print()

t_total0 = time.time()

for i, k in enumerate(K_GRID_NEW):
    name = f"ssmc_k{k:g}"
    csv_path = f"{LOG_DIR}/{name}/version_0/metrics.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        val = df.dropna(subset=["val_reward_mean"])
        if len(val) > 0 and int(val["step"].max()) >= TOTAL_STEPS - 1:
            print(f"[{i + 1}/{len(K_GRID_NEW)}] {name}: already complete, skipping.")
            continue

    for v in range(3):
        p = f"{LOG_DIR}/{name}/version_{v}"
        if os.path.exists(p):
            shutil.rmtree(p)
    p = f"{CKPT_DIR}/{name}"
    if os.path.exists(p):
        shutil.rmtree(p)

    print(f"\n{'=' * 70}")
    print(f"  [{i + 1}/{len(K_GRID_NEW)}] {name}  (k = {k})")
    print(f"{'=' * 70}")

    t0 = time.time()
    vm = ValueNetwork(D, bias=bias_val)
    model = OnPolicyValue(
        base_score_module=base_drift,
        value_module=vm,
        reward_function=reward_fn,
        dim=D,
        a=a,
        lr=LR,
        loss_type="quad",
        analytical_value_fn=anal_fn,
        ema_decay=EMA_DECAY,
    )
    ds = OnPolicySMCDataset(
        dim=D,
        drift=base_drift,
        value=model.ema,
        smc_value=make_log_tau(k),
        reward=reward_fn,
        device=DEVICE,
        a=a,
        batch_size=DS_BATCH_ON,
        n_steps=N_STEPS,
        mc_samples_per_step=MC_SAMPLES,
        sampling_method="single_seed_mc",
        off_policy_frac=OFF_POLICY_FRAC,
        generating_function=gmm_sample,
    )
    loader = DataLoader(ds, batch_size=BS)
    tcb = TrajCB(anal_fn)
    ccb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{name}",
        save_last=True,
        save_top_k=1,
        monitor="val_reward_mean",
        mode="max",
        filename="best",
    )
    logger = CSVLogger(LOG_DIR, name=name, version=0)
    trainer = L.Trainer(
        max_steps=TOTAL_STEPS,
        val_check_interval=max(1, TOTAL_STEPS // 30),  # ~30 evals
        callbacks=[ccb, tcb],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)

    elapsed = (time.time() - t0) / 60
    print(f"  Elapsed: {elapsed:.1f} min  (total so far: {(time.time() - t_total0)/60:.1f} min)")

    del model, vm, trainer, loader, ds
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── Gather (incl. reused) + plot ──────────────────────────────────────────
def truncate_to(df, max_step):
    return df[df["step"] <= max_step]


def load_curves(name, csv_path, max_step=TOTAL_STEPS):
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    df = truncate_to(df, max_step)
    val = df.dropna(subset=["val_reward_mean"])
    mae = df.dropna(subset=["traj_avg_mae_guided"])
    return val, mae


configs = [
    {"name": "off-policy",         "csv": PRIOR_OFF_CSV, "color": "#1f77b4", "ls": "--", "lw": 2.0},
]
for k in K_GRID_NEW:
    configs.append({
        "name": f"ssmc k={k:g}",
        "csv": f"{LOG_DIR}/ssmc_k{k:g}/version_0/metrics.csv",
        "color": None,
        "ls": "-",
        "lw": 1.6,
    })
configs.append({"name": "ssmc k=1", "csv": PRIOR_K1_CSV, "color": "#d62728", "ls": "-", "lw": 2.0})

# Assign colors for the new k values (gradient blue→red as k grows).
new_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(K_GRID_NEW)))
ci = 0
for cfg in configs:
    if cfg["color"] is None:
        cfg["color"] = new_colors[ci]
        ci += 1


print(f"\n{'=' * 70}\n  RESULTS (truncated to step ≤ {TOTAL_STEPS},  E_OPT = {E_OPT:.4f})\n{'=' * 70}")
summary = {}
for cfg in configs:
    val, mae = load_curves(cfg["name"], cfg["csv"])
    if val is None or len(val) == 0:
        print(f"  {cfg['name']:<20}  (no metrics)")
        continue
    best = float(val["val_reward_mean"].max())
    final = float(val["val_reward_mean"].iloc[-1])
    final_mae = float(mae["traj_avg_mae_guided"].iloc[-1]) if len(mae) else float("nan")
    summary[cfg["name"]] = {"best": best, "final": final, "guided_mae": final_mae}
    print(f"  {cfg['name']:<20} best={best:>7.3f}  final={final:>7.3f}  guided_MAE={final_mae:>6.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"single_seed_mc k-sweep on moons  (log_tau = k·h, mc={MC_SAMPLES}, "
    f"steps≤{TOTAL_STEPS})",
    fontsize=12,
    fontweight="bold",
)

# Left: terminal reward
ax = axes[0]
ax.set_title("Terminal reward vs training step")
for cfg in configs:
    val, _ = load_curves(cfg["name"], cfg["csv"])
    if val is None or len(val) == 0:
        continue
    ax.plot(val["step"], val["val_reward_mean"],
            color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"], label=cfg["name"])
ax.axhline(E_OPT, color="black", ls=":", alpha=0.5, label=f"E_opt = {E_OPT:.2f}")
ax.set_xlabel("training step")
ax.set_ylabel("avg terminal reward")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: value MAE on guided trajectories (log scale)
ax = axes[1]
ax.set_title("Value error vs oracle (MAE on guided trajectories)")
for cfg in configs:
    _, mae = load_curves(cfg["name"], cfg["csv"])
    if mae is None or len(mae) == 0:
        continue
    ax.plot(mae["step"], mae["traj_avg_mae_guided"],
            color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"], label=cfg["name"])
ax.set_xlabel("training step")
ax.set_ylabel("MAE  |V_θ − V*|")
ax.set_yscale("log")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
out = "notebooks/ssmc_k_sweep.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved plot: {out}")

with open("notebooks/ssmc_k_sweep_results.json", "w") as f:
    json.dump(
        {
            "config": {
                "TOTAL_STEPS": TOTAL_STEPS,
                "BS": BS,
                "LR": LR,
                "MC_SAMPLES": MC_SAMPLES,
                "OFF_POLICY_FRAC": OFF_POLICY_FRAC,
                "K_GRID_NEW": K_GRID_NEW,
                "E_OPT": E_OPT,
            },
            "summary": summary,
        },
        f,
        indent=2,
    )

print(f"\nTotal elapsed: {(time.time() - t_total0)/60:.1f} min")
print("Done.")
