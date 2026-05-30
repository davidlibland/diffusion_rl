#!/usr/bin/env python3
"""Test the trajectory-correlation hypothesis.

Hypothesis: in vanilla SSMC training each mini-batch of 256 samples is drawn
from only ~4 trajectories (DS_BATCH_ON=ceil(32/mc)=4 with mc=10), so the
gradient estimates are heavily correlated batch-to-batch. If we instead
pre-generate a very large SSMC buffer and shuffle it globally, each batch
of 256 will contain samples from ~256 distinct trajectories — breaking the
within-batch correlation entirely.

If correlation is the bottleneck, this should close most of the gap to
off-policy. If not, the residual gap is from within-trajectory target
bias / target variance.

Configs (all 1000 steps, BS=256, lr=1e-3, mc=10, n_steps=100):
  - off-policy                 (REUSED)
  - ssmc k=0    (vanilla)      (REUSED)
  - ssmc k=0.1  (vanilla)      (REUSED)
  - ssmc k=0    + shuffle-buf  (NEW)
  - ssmc k=0.1  + shuffle-buf  (NEW)

Buffer = N_TRAJ_BIG single_seed_mc trajectories generated up-front.
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

from diffusion_rl.models.on_policy import OnPolicyValue, single_seed_mc
from diffusion_rl.modules.resnet_mlp import ValueNetwork


# ── Setup (matches earlier sweeps) ────────────────────────────────────────
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
            ) / d
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
    if k == 0.0:
        def _zero(x, t):
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        return _zero
    def _scaled(x, t):
        return k * reward_fn(x)
    return _scaled


def zero_value(x, t):
    return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


# ── Constants ──────────────────────────────────────────────────────────────
LOG_DIR = "lightning_logs/ssmc_shuffled_buffer"
CKPT_DIR = "checkpoints/ssmc_shuffled_buffer"
PRIOR_OFF_CSV = "lightning_logs/ssmc_vs_offpolicy/offpolicy/version_0/metrics.csv"
PRIOR_K_CSV = "lightning_logs/ssmc_k_sweep/ssmc_k{k:g}/version_0/metrics.csv"

TOTAL_STEPS = 1000
BS = 256
LR = 1e-3
N_STEPS = 100
MC_SAMPLES = 10
N_TRAJ_BIG = 2000   # 2000 × 100 = 200K samples per buffer

with open("experiments/common/analytical_target.json") as f:
    _at = json.load(f)
E_OPT = _at["E_opt"]

all_rewards = reward_fn(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (
    torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r
).item()

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


# ── Generate the shuffled buffer for a given k ─────────────────────────────
@torch.no_grad()
def generate_buffer(k, n_traj, chunk=200):
    """Generate a large shuffled (x, t, target) buffer from single_seed_mc."""
    print(f"  Generating {n_traj} trajectories for k={k} on {DEVICE}...")
    t0 = time.time()
    xs, ts, ys = [], [], []
    for start in range(0, n_traj, chunk):
        b = min(chunk, n_traj - start)
        all_x, all_t, all_tgt = single_seed_mc(
            drift=base_drift,
            value=zero_value,
            log_tau=make_log_tau(k),
            h=reward_fn,
            a=a,
            batch_size=b,
            mc_samples=MC_SAMPLES,
            dim=D,
            n_steps=N_STEPS,
            device=DEVICE,
        )
        xs.append(all_x.cpu())
        ts.append(all_t.cpu())
        ys.append(all_tgt.cpu())
    x_buf = torch.cat(xs, dim=0)        # (N_TRAJ * N_STEPS, D)
    t_buf = torch.cat(ts, dim=0)
    y_buf = torch.cat(ys, dim=0)

    # Global shuffle (kills within-batch correlation when DataLoader pulls).
    perm = torch.randperm(x_buf.shape[0])
    x_buf = x_buf[perm]
    t_buf = t_buf[perm]
    y_buf = y_buf[perm]

    print(f"    buffer shape x={tuple(x_buf.shape)}  ({(time.time()-t0)/60:.1f} min)")
    return x_buf, t_buf.unsqueeze(-1), y_buf.unsqueeze(-1)


# ── Train one config from a pre-built buffer ──────────────────────────────
def train_from_buffer(name, k, x_buf, t_buf, y_buf):
    csv_path = f"{LOG_DIR}/{name}/version_0/metrics.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        val = df.dropna(subset=["val_reward_mean"])
        if len(val) > 0 and int(val["step"].max()) >= TOTAL_STEPS - 1:
            print(f"  {name}: already complete, skipping.")
            return

    for v in range(3):
        p = f"{LOG_DIR}/{name}/version_{v}"
        if os.path.exists(p):
            shutil.rmtree(p)
    p = f"{CKPT_DIR}/{name}"
    if os.path.exists(p):
        shutil.rmtree(p)

    print(f"\n  [training] {name}  (k = {k})")
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
        ema_decay=0.999,
    )

    # OnPolicyValue.training_step expects (y, x, t).
    ds = TensorDataset(y_buf, x_buf, t_buf)
    loader = DataLoader(ds, batch_size=BS, shuffle=True, drop_last=True)

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
        val_check_interval=max(1, TOTAL_STEPS // 30),
        callbacks=[ccb, tcb],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    print(f"    elapsed: {(time.time() - t0)/60:.1f} min")
    del model, vm, trainer, loader, ds
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── Run experiments ────────────────────────────────────────────────────────
print(f"E_OPT = {E_OPT:.4f}")
print(f"Device: {DEVICE}")
print(f"BS={BS}  LR={LR}  TOTAL_STEPS={TOTAL_STEPS}  N_STEPS={N_STEPS}")
print(f"Buffer: N_TRAJ={N_TRAJ_BIG}  → {N_TRAJ_BIG * N_STEPS} samples per k")
print()

t_total0 = time.time()
for k in [0.0, 0.1]:
    name = f"shuf_k{k:g}"
    print(f"\n=== k = {k} ===")
    x_buf, t_buf, y_buf = generate_buffer(k, N_TRAJ_BIG)
    train_from_buffer(name, k, x_buf, t_buf, y_buf)
    del x_buf, t_buf, y_buf
    gc.collect()

print(f"\nTotal elapsed: {(time.time() - t_total0)/60:.1f} min")


# ── Plot: compare shuffled-buffer SSMC vs vanilla SSMC vs off-policy ──────
def truncate_to(df, max_step):
    return df[df["step"] <= max_step]


def load_curves(csv_path, max_step=TOTAL_STEPS):
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    df = truncate_to(df, max_step)
    val = df.dropna(subset=["val_reward_mean"])
    mae = df.dropna(subset=["traj_avg_mae_guided"])
    return val, mae


configs = [
    {"name": "off-policy",         "csv": PRIOR_OFF_CSV,                        "color": "#1f77b4", "ls": "--", "lw": 2.0},
    {"name": "ssmc k=0  vanilla",  "csv": PRIOR_K_CSV.format(k=0.0),            "color": "#7f7f7f", "ls": "-",  "lw": 1.4},
    {"name": "ssmc k=0  shuffle",  "csv": f"{LOG_DIR}/shuf_k0/version_0/metrics.csv","color": "#2ca02c", "ls": "-", "lw": 2.0},
    {"name": "ssmc k=0.1 vanilla", "csv": PRIOR_K_CSV.format(k=0.1),            "color": "#bcbd22", "ls": "-",  "lw": 1.4},
    {"name": "ssmc k=0.1 shuffle", "csv": f"{LOG_DIR}/shuf_k0.1/version_0/metrics.csv","color": "#9467bd", "ls": "-", "lw": 2.0},
]

print(f"\n{'=' * 70}\n  RESULTS  (truncated to step ≤ {TOTAL_STEPS})\n{'=' * 70}")
summary = {}
for cfg in configs:
    val, mae = load_curves(cfg["csv"])
    if val is None or len(val) == 0:
        print(f"  {cfg['name']:<22}  (no metrics)")
        continue
    best = float(val["val_reward_mean"].max())
    final = float(val["val_reward_mean"].iloc[-1])
    final_mae = float(mae["traj_avg_mae_guided"].iloc[-1]) if len(mae) else float("nan")
    summary[cfg["name"]] = {"best": best, "final": final, "guided_mae": final_mae}
    print(f"  {cfg['name']:<22} best={best:>7.3f}  final={final:>7.3f}  guided_MAE={final_mae:>6.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"single_seed_mc with pre-shuffled buffer (N_TRAJ={N_TRAJ_BIG}, mc={MC_SAMPLES}, steps≤{TOTAL_STEPS})",
    fontsize=12, fontweight="bold",
)

ax = axes[0]
ax.set_title("Terminal reward vs training step")
for cfg in configs:
    val, _ = load_curves(cfg["csv"])
    if val is None or len(val) == 0:
        continue
    ax.plot(val["step"], val["val_reward_mean"],
            color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"], label=cfg["name"])
ax.axhline(E_OPT, color="black", ls=":", alpha=0.5, label=f"E_opt = {E_OPT:.2f}")
ax.set_xlabel("training step")
ax.set_ylabel("avg terminal reward")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.set_title("Value error vs oracle (MAE on guided trajectories)")
for cfg in configs:
    _, mae = load_curves(cfg["csv"])
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
out = "experiments/misc/2026-05-28_ssmc_sweeps/ssmc_shuffled_buffer.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")

with open("experiments/misc/2026-05-28_ssmc_sweeps/ssmc_shuffled_buffer_results.json", "w") as f:
    json.dump(
        {
            "config": {
                "TOTAL_STEPS": TOTAL_STEPS, "BS": BS, "LR": LR,
                "MC_SAMPLES": MC_SAMPLES, "N_TRAJ_BIG": N_TRAJ_BIG,
                "E_OPT": E_OPT,
            },
            "summary": summary,
        },
        f, indent=2,
    )
print("Done.")
