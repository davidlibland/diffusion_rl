#!/usr/bin/env python3
"""Re-run small-BS cells from bs_sweep.py with balanced-coverage MAE.

For each (method, BS in {1,4,16}, seed):
  - train with the same recipe as bs_sweep.py
  - additionally log MAE on TWO fixed eval sets per run:
      off_dist_mae   : eval points from off-policy Brownian-bridge distribution
      ssmc_dist_mae  : eval points from SSMC trajectory distribution (the same
                       distribution used for SSMC training: mc=10, n_steps=30
                       random, smc_value = 0.1 * h(x) * t)
  - save best (by val_reward_mean) checkpoint
After training, report each metric at the val step where val_reward_mean was max.
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
from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork


# ── Shared setup ──────────────────────────────────────────────────────────
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scaler = StandardScaler(); X = scaler.fit_transform(X)
clf = GaussianMixture(n_components=100, covariance_type="spherical"); clf.fit(X)
_means = torch.from_numpy(clf.means_).double()
_sigma2 = torch.from_numpy(clf.covariances_).double()
_weights = torch.from_numpy(clf.weights_).double()
_sigmas = torch.sqrt(_sigma2)[:, None]; _weights_col = _weights[:, None]

D = 2; a = 1.0; c = torch.tensor([1.0, 0.0])
means_np = clf.means_; sigmas_np = np.sqrt(clf.covariances_); weights_np = clf.weights_


class AV(nn.Module):
    def __init__(self, means, sigma2, weights, a=1.0, c_=None, D_=2):
        super().__init__()
        if c_ is None: c_ = torch.tensor([1.0, 0.0])
        self.register_buffer("means", means.float())
        self.register_buffer("sigma2", sigma2.float())
        self.register_buffer("weights", weights.float())
        self.a = a; self.D = D_; self.register_buffer("c", c_.float())

    def _log_Z(self, m, v):
        cc = self.c.double(); d = 1 + 20 * v
        return (
            -self.D / 2 * torch.log(d)
            + (-10 * (m**2).sum(-1) + 20 * (m * cc).sum(-1) + 200 * v * (cc**2).sum()) / d
            - 10 * (cc**2).sum()
        )

    def forward(self, x, t):
        x = x.double(); t = t.double().reshape(-1)
        if t.numel() == 1: t = t.expand(x.shape[0])
        t_ = t[:, None]
        m = self.means.double(); s2 = self.sigma2.double(); w = self.weights.double()
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
    ts = ts.reshape(-1, 1); xt_ = xt[..., None]
    means_ = _means.float().to(xt).T[None, ...]; ts_ = ts[..., None]
    sigmas_ = _sigmas.float().to(xt).T; weights_ = _weights_col.float().to(xt).T
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


def gmm_sample(n):
    k_ = np.random.choice(len(weights_np), size=n, p=weights_np)
    return means_np[k_] + sigmas_np[k_, np.newaxis] * np.random.randn(n, D)


def zero_value(x, t):
    return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


def smc_value_fn(x, t):
    return 0.1 * reward_fn(x) * t.reshape(-1)


class OnPolicyValueLive(OnPolicyValue):
    def drift(self, x, t, beta=1, use_ema=False):
        return super().drift(x, t, beta=beta, use_ema=use_ema)


# ── Constants ──────────────────────────────────────────────────────────────
LOG_DIR = "lightning_logs/bs_sweep_balanced"
CKPT_DIR = "checkpoints/bs_sweep_balanced"

TOTAL_STEPS = 1000
LR = 1e-3
N_STEPS_TRAIN = 30
N_STEPS_VAL = 100
MC_SAMPLES = 10
N_TRAJ_BUFFER = 2000
N_EVAL = 2048               # size of each balanced-MAE eval set
N_SEEDS = 5

BS_VALUES = [1, 4, 16]

with open("experiments/common/analytical_target.json") as f:
    E_OPT = json.load(f)["E_opt"]
all_rewards = reward_fn(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()
val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


# ── Eval-set builders ─────────────────────────────────────────────────────
@torch.no_grad()
def make_off_dist_eval(n=N_EVAL):
    """Sample (x, t) from off-policy Brownian-bridge distribution."""
    x1 = torch.from_numpy(gmm_sample(n)).float()
    t = torch.rand(n)
    eps = torch.randn(n, D)
    x = t[:, None] * x1 + torch.sqrt(2 * a * t * (1 - t))[:, None] * eps
    return x, t


@torch.no_grad()
def make_ssmc_dist_eval(n=N_EVAL, chunk=400):
    """Sample (x, t) from the SSMC trajectory distribution
    (smc_value = 0.1·h·t, mc=10, n_steps=30 random, no t=0)."""
    # Each trajectory yields N_STEPS_TRAIN samples.  Need n_traj * N_STEPS_TRAIN >= n.
    n_traj = (n + N_STEPS_TRAIN - 1) // N_STEPS_TRAIN
    xs, ts = [], []
    for start in range(0, n_traj, chunk):
        b = min(chunk, n_traj - start)
        ax, at, _ = single_seed_mc(
            drift=base_drift,
            value=zero_value,
            log_tau=smc_value_fn,
            h=reward_fn,
            a=a,
            batch_size=b,
            mc_samples=MC_SAMPLES,
            dim=D,
            n_steps=N_STEPS_TRAIN,
            device=DEVICE,
            random_t=True,
            include_t_zero=False,
        )
        xs.append(ax.cpu()); ts.append(at.cpu())
    x_all = torch.cat(xs, 0); t_all = torch.cat(ts, 0)
    idx = torch.randperm(x_all.shape[0])[:n]
    return x_all[idx], t_all[idx]


# ── Callbacks ─────────────────────────────────────────────────────────────
class BalancedMAECallback(Callback):
    """Logs MAE / bias of pl.value_module vs analytical V on a fixed eval set."""
    def __init__(self, eval_x, eval_t, af, name: str):
        super().__init__()
        self.eval_x = eval_x.detach()
        self.eval_t = eval_t.detach()
        self.af = af
        self.name = name

    def on_validation_batch_end(self, trainer, pl, outputs, batch, bi):
        if bi > 0:
            return
        x = self.eval_x.to(pl.device)
        t = self.eval_t.to(pl.device)
        with torch.no_grad():
            v_pred = pl.value_module(x, t)
            v_anal = self.af(x, t)
        err = v_pred - v_anal
        pl.log(f"{self.name}_mae", err.abs().mean(), prog_bar=False)
        pl.log(f"{self.name}_bias", err.mean(), prog_bar=False)


class TrajCB(Callback):
    """Same TrajCB as in bs_sweep.py — guided-rollout MAE."""
    def __init__(self, af, n=256, ns=N_STEPS_VAL):
        super().__init__(); self.af = af; self.n = n; self.ns = ns

    def on_validation_batch_end(self, trainer, pl, outputs, batch, bi):
        if bi > 0:
            return
        dim = pl.hparams.dim; dev = pl.device; n = self.n; dt = 1.0 / self.ns
        for beta, label in [(0, "base"), (1, "guided")]:
            x = torch.zeros(n, dim, device=dev)
            ax, at = [x], [torch.zeros(n, device=dev)]
            dfn = partial(pl.drift, beta=beta)
            for st in torch.linspace(0, 1, self.ns + 1, device=dev)[:-1]:
                tv = st.expand(n)
                dx = dfn(x, tv) * dt
                db = sqrt(2 * pl.a * dt) * torch.randn_like(x)
                x = x + dx + db
                ax.append(x); at.append(torch.full((n,), float(st) + dt, device=dev))
            ax = torch.cat(ax); at = torch.cat(at)
            with torch.no_grad():
                vp = pl.value_module(ax, at); va = self.af(ax, at)
            err = vp - va
            pl.log(f"traj_avg_mae_{label}", err.abs().mean(), prog_bar=False)


# ── SSMC buffer + run-one ─────────────────────────────────────────────────
@torch.no_grad()
def generate_ssmc_buffer(n_traj=N_TRAJ_BUFFER, chunk=400):
    xs, ts, ys = [], [], []
    for start in range(0, n_traj, chunk):
        b = min(chunk, n_traj - start)
        ax, at, ay = single_seed_mc(
            drift=base_drift, value=zero_value, log_tau=smc_value_fn, h=reward_fn,
            a=a, batch_size=b, mc_samples=MC_SAMPLES, dim=D, n_steps=N_STEPS_TRAIN,
            device=DEVICE, random_t=True, include_t_zero=False,
        )
        xs.append(ax.cpu()); ts.append(at.cpu()); ys.append(ay.cpu())
    x_buf = torch.cat(xs, 0); t_buf = torch.cat(ts, 0); y_buf = torch.cat(ys, 0)
    perm = torch.randperm(x_buf.shape[0])
    return x_buf[perm], t_buf[perm].unsqueeze(-1), y_buf[perm].unsqueeze(-1)


def run_one(method, bs, seed, eval_off, eval_ssmc):
    name = f"{method}_bs{bs:03d}_seed{seed:02d}"
    csv_path = f"{LOG_DIR}/{name}/version_0/metrics.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        v = df.dropna(subset=["val_reward_mean"])
        if len(v) > 0 and int(v["step"].max()) >= TOTAL_STEPS - 1:
            print(f"  {name}: already complete, skipping.", flush=True)
            return
    for vv in range(3):
        p = f"{LOG_DIR}/{name}/version_{vv}"
        if os.path.exists(p):
            shutil.rmtree(p)
    p = f"{CKPT_DIR}/{name}"
    if os.path.exists(p):
        shutil.rmtree(p)

    L.seed_everything(seed, workers=True)
    print(f"\n=== {method}  bs={bs}  seed={seed} ===", flush=True)
    t0 = time.time()

    vm = ValueNetwork(D, bias=bias_val)
    if method == "ssmc":
        model = OnPolicyValueLive(
            base_score_module=base_drift, value_module=vm, reward_function=reward_fn,
            dim=D, a=a, lr=LR, loss_type="quad",
            analytical_value_fn=anal_fn, ema_decay=0.999,
        )
        x_buf, t_buf, y_buf = generate_ssmc_buffer()
        ds = TensorDataset(y_buf, x_buf, t_buf)
        loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    elif method == "off":
        model = OffPolicyValue(
            base_score_module=base_drift, value_module=vm, reward_function=reward_fn,
            dim=D, a=a, lr=LR, loss_type="quad",
            analytical_value_fn=anal_fn,
        )
        ds = InterpolatingNumpyDataset(
            generating_function=gmm_sample, a=a, batch_size=1024
        )
        loader = DataLoader(ds, batch_size=bs)
    else:
        raise ValueError(method)

    cbs = [
        TrajCB(anal_fn),
        BalancedMAECallback(eval_off[0], eval_off[1], anal_fn, name="off_dist"),
        BalancedMAECallback(eval_ssmc[0], eval_ssmc[1], anal_fn, name="ssmc_dist"),
        ModelCheckpoint(
            dirpath=f"{CKPT_DIR}/{name}",
            save_last=False, save_top_k=1,
            monitor="val_reward_mean", mode="max", filename="best",
        ),
    ]
    logger = CSVLogger(LOG_DIR, name=name, version=0)
    tr = L.Trainer(
        max_steps=TOTAL_STEPS, val_check_interval=max(1, TOTAL_STEPS // 30),
        callbacks=cbs, logger=logger,
        enable_checkpointing=True, enable_progress_bar=False,
    )
    tr.fit(model, loader, val_dataloaders=val_loader)
    print(f"  {method} bs={bs} seed={seed} elapsed: {(time.time() - t0)/60:.1f} min",
          flush=True)
    del model, vm, tr, loader, ds
    if method == "ssmc":
        del x_buf, t_buf, y_buf
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── Build shared eval sets (one pair per seed so paths see the same eval) ─
# Reuse a fixed seed-independent eval pair across all (method, BS, seed) — the
# eval distribution doesn't depend on the trained network, so this is fine and
# makes comparisons cleaner.
print("Building eval sets...", flush=True)
torch.manual_seed(12345)
np.random.seed(12345)
eval_off = make_off_dist_eval()
eval_ssmc = make_ssmc_dist_eval()
print(f"  off_dist eval:  x {tuple(eval_off[0].shape)}  t {tuple(eval_off[1].shape)}")
print(f"  ssmc_dist eval: x {tuple(eval_ssmc[0].shape)}  t {tuple(eval_ssmc[1].shape)}",
      flush=True)


# ── Run ───────────────────────────────────────────────────────────────────
print(f"\nE_OPT = {E_OPT:.4f}")
print(f"Device: {DEVICE}")
print(f"BS grid: {BS_VALUES}  seeds: {N_SEEDS}", flush=True)
t_total0 = time.time()
for method in ["off", "ssmc"]:
    for bs in BS_VALUES:
        for s in range(N_SEEDS):
            run_one(method, bs, s, eval_off, eval_ssmc)
            print(f"  total elapsed: {(time.time() - t_total0)/60:.1f} min", flush=True)


# ── Aggregate at best-val-reward step ─────────────────────────────────────
def best_step_metrics(csv_path, metrics, monitor="val_reward_mean"):
    """Return dict of metric values at the step where `monitor` is max."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    v = df.dropna(subset=[monitor])
    if len(v) == 0:
        return None
    best_step = int(v.loc[v[monitor].idxmax(), "step"])
    out = {"best_step": best_step, monitor: float(v[monitor].max())}
    for m in metrics:
        sub = df.dropna(subset=[m])
        if len(sub) == 0:
            out[m] = float("nan")
            continue
        # value at the step closest to (and ≤) best_step
        sub2 = sub[sub["step"] <= best_step]
        if len(sub2) == 0:
            sub2 = sub
        out[m] = float(sub2.iloc[-1][m])
    return out


METRICS = ["off_dist_mae", "off_dist_bias",
           "ssmc_dist_mae", "ssmc_dist_bias",
           "traj_avg_mae_guided"]
agg = {}
for method in ["off", "ssmc"]:
    for bs in BS_VALUES:
        cells = []
        for s in range(N_SEEDS):
            csv = f"{LOG_DIR}/{method}_bs{bs:03d}_seed{s:02d}/version_0/metrics.csv"
            r = best_step_metrics(csv, METRICS)
            if r is not None:
                cells.append(r)
        agg[(method, bs)] = cells


def mean_sd(vals):
    a = np.array(vals, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0)


print("\n" + "=" * 100)
print(f"{'method':>6} {'BS':>3} {'n':>3} | "
      f"{'best_R':>8} | "
      f"{'off_dist_mae':>14} | {'ssmc_dist_mae':>14} | {'traj_mae_g':>10}")
print("=" * 100)
for method in ["off", "ssmc"]:
    for bs in BS_VALUES:
        cells = agg[(method, bs)]
        n = len(cells)
        if n == 0:
            continue
        m_r, s_r = mean_sd([c["val_reward_mean"] for c in cells])
        m_o, s_o = mean_sd([c["off_dist_mae"] for c in cells])
        m_s, s_s = mean_sd([c["ssmc_dist_mae"] for c in cells])
        m_t, s_t = mean_sd([c["traj_avg_mae_guided"] for c in cells])
        print(f"{method:>6} {bs:>3} {n:>3} | "
              f"{m_r:>5.2f}±{s_r:>4.2f} | "
              f"{m_o:>5.2f}±{s_o:>4.2f}{'':>2} | "
              f"{m_s:>5.2f}±{s_s:>4.2f}{'':>2} | "
              f"{m_t:>4.2f}±{s_t:>4.2f}")
print("=" * 100)


# ── Plot: 2×3 grid showing each method on each eval distribution ──────────
fig, axes = plt.subplots(2, len(BS_VALUES), figsize=(5 * len(BS_VALUES), 9), sharey="row")
fig.suptitle(
    "Balanced-coverage MAE at best-reward checkpoint — small-BS slice "
    f"(n={N_SEEDS} seeds / cell)",
    fontsize=12, fontweight="bold",
)
OFF_C = "#1f77b4"; SSMC_C = "#ff7f0e"

for col, bs in enumerate(BS_VALUES):
    # off_dist eval
    ax = axes[0, col]
    ax.set_title(f"BS = {bs}  ·  evaluated on off-policy distribution")
    vals = {
        "off→off": [c["off_dist_mae"] for c in agg[("off", bs)]],
        "ssmc→off": [c["off_dist_mae"] for c in agg[("ssmc", bs)]],
    }
    pos = [0, 1]; labels = list(vals.keys())
    ax.bar(pos[0], np.mean(vals["off→off"]), yerr=np.std(vals["off→off"], ddof=1),
           color=OFF_C, capsize=4)
    ax.bar(pos[1], np.mean(vals["ssmc→off"]), yerr=np.std(vals["ssmc→off"], ddof=1),
           color=SSMC_C, capsize=4)
    for i, k in enumerate(labels):
        for v in vals[k]:
            ax.scatter(i, v, color="k", s=12, alpha=0.5, zorder=3)
    ax.set_xticks(pos); ax.set_xticklabels(labels)
    if col == 0:
        ax.set_ylabel("off_dist MAE")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")

    # ssmc_dist eval
    ax = axes[1, col]
    ax.set_title(f"BS = {bs}  ·  evaluated on SSMC distribution")
    vals = {
        "off→ssmc": [c["ssmc_dist_mae"] for c in agg[("off", bs)]],
        "ssmc→ssmc": [c["ssmc_dist_mae"] for c in agg[("ssmc", bs)]],
    }
    pos = [0, 1]; labels = list(vals.keys())
    ax.bar(pos[0], np.mean(vals["off→ssmc"]), yerr=np.std(vals["off→ssmc"], ddof=1),
           color=OFF_C, capsize=4)
    ax.bar(pos[1], np.mean(vals["ssmc→ssmc"]), yerr=np.std(vals["ssmc→ssmc"], ddof=1),
           color=SSMC_C, capsize=4)
    for i, k in enumerate(labels):
        for v in vals[k]:
            ax.scatter(i, v, color="k", s=12, alpha=0.5, zorder=3)
    ax.set_xticks(pos); ax.set_xticklabels(labels)
    if col == 0:
        ax.set_ylabel("ssmc_dist MAE")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
out = "experiments/misc/2026-05-28_bs_sweep/bs_sweep_balanced.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")


# ── Save JSON ─────────────────────────────────────────────────────────────
summary = {
    "config": {
        "TOTAL_STEPS": TOTAL_STEPS, "LR": LR,
        "N_STEPS_TRAIN": N_STEPS_TRAIN, "N_STEPS_VAL": N_STEPS_VAL,
        "MC_SAMPLES": MC_SAMPLES, "N_TRAJ_BUFFER": N_TRAJ_BUFFER,
        "N_EVAL": N_EVAL, "BS_VALUES": BS_VALUES, "N_SEEDS": N_SEEDS,
        "smc_value": "0.1 * h(x) * t", "E_OPT": E_OPT,
        "include_t_zero": False, "random_t": True,
    },
    "by_cell": {
        f"{method}_bs={bs}": agg[(method, bs)]
        for method in ["off", "ssmc"] for bs in BS_VALUES
    },
}
with open("experiments/misc/2026-05-28_bs_sweep/bs_sweep_balanced_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nTotal elapsed: {(time.time() - t_total0)/60:.1f} min")
print("Done.")
