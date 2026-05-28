#!/usr/bin/env python3
"""Single SSMC k=0 1-per-traj seed with use_ema=False in drift.

Tests the hypothesis that the residual reward gap to off-policy is caused
by validation rollouts using the EMA shadow (decay=0.999, ~37% initial
weight at step 1000) instead of the live value module.

Subclasses OnPolicyValue and flips `drift`'s default to use_ema=False.
Both validation_step's reward rollout and the TrajCB guided-trajectory
MAE callback then use the live value module.

If reward jumps to off-policy levels (~-4.5) and guided MAE drops, that
confirms EMA-in-rollout was the bottleneck.
"""

import gc
import json
import os
import shutil
import time
from functools import partial
from math import sqrt

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


# ── Setup (matches earlier sweeps) ─────────────────────────────────────────
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


def log_tau_zero(x, t):
    return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


def zero_value(x, t):
    return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


# ── Subclass: drift defaults to use_ema=False ──────────────────────────────
class OnPolicyValueLive(OnPolicyValue):
    """Same as OnPolicyValue but `drift` uses the live value_module, not EMA."""

    def drift(self, x, t, beta=1, use_ema=False):
        return super().drift(x, t, beta=beta, use_ema=use_ema)


# ── Constants ──────────────────────────────────────────────────────────────
LOG_DIR = "lightning_logs/ssmc_1pt_no_ema"
CKPT_DIR = "checkpoints/ssmc_1pt_no_ema"

TOTAL_STEPS = 1000
BS = 256
LR = 1e-3
N_STEPS = 100
MC_SAMPLES = 10
N_TRAJ_BIG = 256_000
SEED = 0

with open("notebooks/analytical_target.json") as f:
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
            dfn = partial(pl.drift, beta=beta)   # uses subclass default = LIVE
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


# ── Buffer generator (1 sample per trajectory, k=0) ────────────────────────
@torch.no_grad()
def generate_buffer_one_per_traj(n_traj, chunk=800):
    print(f"  Generating {n_traj} trajectories (1 sample/traj kept) on {DEVICE}...")
    t0 = time.time()
    xs, ts, ys = [], [], []
    n_done = 0
    for start in range(0, n_traj, chunk):
        b = min(chunk, n_traj - start)
        all_x, all_t, all_tgt = single_seed_mc(
            drift=base_drift,
            value=zero_value,
            log_tau=log_tau_zero,
            h=reward_fn,
            a=a,
            batch_size=b,
            mc_samples=MC_SAMPLES,
            dim=D,
            n_steps=N_STEPS,
            device=DEVICE,
        )
        x_3d = all_x.reshape(b, N_STEPS, D)
        t_2d = all_t.reshape(b, N_STEPS)
        y_2d = all_tgt.reshape(b, N_STEPS)
        idx = torch.randint(0, N_STEPS, (b,), device=all_x.device)
        rng = torch.arange(b, device=all_x.device)
        xs.append(x_3d[rng, idx].cpu())
        ts.append(t_2d[rng, idx].cpu())
        ys.append(y_2d[rng, idx].cpu())
        n_done += b
        if n_done % (chunk * 10) == 0 or n_done == n_traj:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            eta = (n_traj - n_done) / max(rate, 1e-6)
            print(f"    {n_done:>7}/{n_traj}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
    x_buf = torch.cat(xs)
    t_buf = torch.cat(ts).unsqueeze(-1)
    y_buf = torch.cat(ys).unsqueeze(-1)
    print(f"    buffer: {tuple(x_buf.shape)}  ({(time.time() - t0)/60:.1f} min total)")
    return x_buf, t_buf, y_buf


# ── Run ────────────────────────────────────────────────────────────────────
print(f"E_OPT = {E_OPT:.4f}")
print(f"Device: {DEVICE}")
print(f"BS={BS}  LR={LR}  TOTAL_STEPS={TOTAL_STEPS}  N_TRAJ_BIG={N_TRAJ_BIG}  seed={SEED}")
print("Drift in rollout: LIVE value_module (use_ema=False).")
print()

t_total0 = time.time()

name = f"seed_{SEED:02d}_no_ema"
csv_path = f"{LOG_DIR}/{name}/version_0/metrics.csv"
for v in range(3):
    p = f"{LOG_DIR}/{name}/version_{v}"
    if os.path.exists(p):
        shutil.rmtree(p)
p = f"{CKPT_DIR}/{name}"
if os.path.exists(p):
    shutil.rmtree(p)

L.seed_everything(SEED, workers=True)
x_buf, t_buf, y_buf = generate_buffer_one_per_traj(N_TRAJ_BIG)

print(f"\n  [training] {name}")
t0 = time.time()
vm = ValueNetwork(D, bias=bias_val)
model = OnPolicyValueLive(
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
ds = TensorDataset(y_buf, x_buf, t_buf)
loader = DataLoader(ds, batch_size=BS, shuffle=True, drop_last=True)
tcb = TrajCB(anal_fn)
ccb = ModelCheckpoint(
    dirpath=f"{CKPT_DIR}/{name}",
    save_last=True, save_top_k=1,
    monitor="val_reward_mean", mode="max", filename="best",
)
logger = CSVLogger(LOG_DIR, name=name, version=0)
trainer = L.Trainer(
    max_steps=TOTAL_STEPS,
    val_check_interval=max(1, TOTAL_STEPS // 30),
    callbacks=[ccb, tcb],
    logger=logger,
    enable_checkpointing=True,
    enable_progress_bar=False,
)
trainer.fit(model, loader, val_dataloaders=val_loader)
print(f"    elapsed: {(time.time() - t0)/60:.1f} min")


# ── Aggregate and plot vs prior runs ──────────────────────────────────────
def load_curves(csv_path, max_step=TOTAL_STEPS):
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    df = df[df["step"] <= max_step]
    val = df.dropna(subset=["val_reward_mean"])
    mae = df.dropna(subset=["traj_avg_mae_guided"])
    return val, mae


# This run.
new_val, new_mae = load_curves(csv_path)
new_best = float(new_val["val_reward_mean"].max())
new_final = float(new_val["val_reward_mean"].iloc[-1])
new_final_mae = float(new_mae["traj_avg_mae_guided"].iloc[-1])

# Reference distributions.
with open("notebooks/offpolicy_seeds_results.json") as f:
    off_data = json.load(f)
off_bests = np.array(off_data["best_reward"]["values"])
off_maes = np.array(off_data["final_guided_mae"]["values"])

with open("notebooks/ssmc_1pt_seeds_results.json") as f:
    ema_data = json.load(f)
ema_bests = np.array(ema_data["best_reward"]["values"])
ema_maes = np.array(ema_data["final_guided_mae"]["values"])

print(f"\n{'=' * 72}")
print(f"  RESULTS: SSMC k=0 1-per-traj WITHOUT EMA  (seed {SEED})")
print(f"{'=' * 72}")
print(f"  best reward:        {new_best:>7.3f}")
print(f"  final reward:       {new_final:>7.3f}")
print(f"  final guided MAE:   {new_final_mae:>6.3f}")
print()
print(f"  Off-policy distribution (n={len(off_bests)}):")
print(f"    best reward:      mean={off_bests.mean():>7.3f}  sd={off_bests.std(ddof=1):>5.3f}  "
      f"min={off_bests.min():>7.3f}  max={off_bests.max():>7.3f}")
print(f"    final guided MAE: mean={off_maes.mean():>6.3f}  sd={off_maes.std(ddof=1):>5.3f}  "
      f"min={off_maes.min():>6.3f}  max={off_maes.max():>6.3f}")
print()
print(f"  Prior SSMC 1-per-traj WITH EMA (n={len(ema_bests)}):")
print(f"    best reward:      mean={ema_bests.mean():>7.3f}  sd={ema_bests.std(ddof=1):>5.3f}")
print(f"    final guided MAE: mean={ema_maes.mean():>6.3f}  sd={ema_maes.std(ddof=1):>5.3f}")

z_best = (new_best - off_bests.mean()) / off_bests.std(ddof=1)
z_mae = (new_final_mae - off_maes.mean()) / off_maes.std(ddof=1)
print(f"\n  z-score vs off-policy distribution:")
print(f"    best reward:      z = {z_best:+.2f}")
print(f"    final guided MAE: z = {z_mae:+.2f}")


# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    f"SSMC k=0 1-per-traj  (use_ema={False})  vs off-policy + prior SSMC (with EMA)",
    fontsize=12, fontweight="bold",
)

OFF_C = "#1f77b4"
EMA_C = "#d62728"
NEW_C = "#2ca02c"

# (a) Reward curves
ax = axes[0, 0]
ax.set_title("Reward curves (per seed)")
for s in range(off_data["n_seeds"]):
    val, _ = load_curves(f"lightning_logs/offpolicy_seeds/seed_{s:02d}/version_0/metrics.csv")
    if val is not None:
        ax.plot(val["step"], val["val_reward_mean"], color=OFF_C, alpha=0.4, lw=1.0)
for s in range(ema_data["config"]["n_seeds"]):
    val, _ = load_curves(f"lightning_logs/ssmc_1pt_seeds/seed_{s:02d}/version_0/metrics.csv")
    if val is not None:
        ax.plot(val["step"], val["val_reward_mean"], color=EMA_C, alpha=0.5, lw=1.0)
ax.plot(new_val["step"], new_val["val_reward_mean"], color=NEW_C, lw=2.4, label="ssmc 1pt NO EMA (this run)")
ax.axhline(E_OPT, color="black", ls=":", alpha=0.6, label=f"E_opt={E_OPT:.2f}")
ax.plot([], [], color=OFF_C, lw=2, label=f"off-policy (n={len(off_bests)})")
ax.plot([], [], color=EMA_C, lw=2, label=f"ssmc 1pt WITH EMA (n={len(ema_bests)})")
ax.set_xlabel("training step")
ax.set_ylabel("avg terminal reward")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (b) Best-reward distribution
ax = axes[0, 1]
ax.set_title("Best reward distribution")
ax.hist(off_bests, bins=8, color=OFF_C, alpha=0.55, edgecolor="white",
        label=f"off-policy n={len(off_bests)}")
ax.hist(ema_bests, bins=5, color=EMA_C, alpha=0.55, edgecolor="white",
        label=f"ssmc 1pt EMA n={len(ema_bests)}")
ax.axvline(new_best, color=NEW_C, lw=3, label=f"this run (no EMA) = {new_best:.2f}")
ax.set_xlabel("best val_reward_mean")
ax.set_ylabel("count")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) MAE curves
ax = axes[1, 0]
ax.set_title("Guided value MAE (per seed)")
for s in range(off_data["n_seeds"]):
    _, mae = load_curves(f"lightning_logs/offpolicy_seeds/seed_{s:02d}/version_0/metrics.csv")
    if mae is not None and len(mae) > 0:
        ax.plot(mae["step"], mae["traj_avg_mae_guided"], color=OFF_C, alpha=0.4, lw=1.0)
for s in range(ema_data["config"]["n_seeds"]):
    _, mae = load_curves(f"lightning_logs/ssmc_1pt_seeds/seed_{s:02d}/version_0/metrics.csv")
    if mae is not None and len(mae) > 0:
        ax.plot(mae["step"], mae["traj_avg_mae_guided"], color=EMA_C, alpha=0.5, lw=1.0)
ax.plot(new_mae["step"], new_mae["traj_avg_mae_guided"], color=NEW_C, lw=2.4,
        label="ssmc 1pt NO EMA (this run)")
ax.plot([], [], color=OFF_C, lw=2, label=f"off-policy (n={len(off_bests)})")
ax.plot([], [], color=EMA_C, lw=2, label=f"ssmc 1pt WITH EMA (n={len(ema_bests)})")
ax.set_yscale("log")
ax.set_xlabel("training step")
ax.set_ylabel("MAE  |V_θ − V*|")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which="both")

# (d) Final MAE distribution
ax = axes[1, 1]
ax.set_title("Final guided MAE distribution")
ax.hist(off_maes, bins=8, color=OFF_C, alpha=0.55, edgecolor="white",
        label=f"off-policy n={len(off_maes)}")
ax.hist(ema_maes, bins=5, color=EMA_C, alpha=0.55, edgecolor="white",
        label=f"ssmc 1pt EMA n={len(ema_maes)}")
ax.axvline(new_final_mae, color=NEW_C, lw=3, label=f"this run (no EMA) = {new_final_mae:.2f}")
ax.set_xlabel("final guided MAE")
ax.set_ylabel("count")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "notebooks/ssmc_1pt_no_ema.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nSaved: {out}")

with open("notebooks/ssmc_1pt_no_ema_results.json", "w") as f:
    json.dump(
        {
            "config": {
                "TOTAL_STEPS": TOTAL_STEPS, "BS": BS, "LR": LR,
                "MC_SAMPLES": MC_SAMPLES, "N_TRAJ_BIG": N_TRAJ_BIG,
                "E_OPT": E_OPT, "use_ema": False, "seed": SEED,
            },
            "best_reward": new_best,
            "final_reward": new_final,
            "final_guided_mae": new_final_mae,
            "z_best_vs_offpolicy": float(z_best),
            "z_mae_vs_offpolicy": float(z_mae),
        },
        f, indent=2,
    )

print(f"\nTotal elapsed: {(time.time() - t_total0)/60:.1f} min")
print("Done.")
