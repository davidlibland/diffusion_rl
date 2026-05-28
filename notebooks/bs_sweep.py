#!/usr/bin/env python3
"""Batch-size sweep, off-policy vs SSMC with smc_value = 0.1·h(x)·t.

SSMC settings:
  - mc_samples = 10
  - n_steps    = 30 (random_t=True → 30 sorted U(0,1) points + the endpoints)
  - include_t_zero=False
  - shuffled pre-generated buffer (2000 trajectories ≈ 60K samples per seed)

BS ∈ {1, 4, 16, 64, 256}.  5 seeds per (method, BS) cell.
For BS=256 off-policy we reuse the first 5 seeds from
``notebooks/offpolicy_seeds_results.json`` (existing) rather than re-run.

Same max_steps=1000 across all BS — controlled gradient-update count.
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
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.models.on_policy import OnPolicyValue, single_seed_mc
from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork


# ── Setup ─────────────────────────────────────────────────────────────────
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
    """smc_value(x, t) = 0.1 * h(x) * t  (fixed; no trained network)."""
    return 0.1 * reward_fn(x) * t.reshape(-1)


class OnPolicyValueLive(OnPolicyValue):
    def drift(self, x, t, beta=1, use_ema=False):
        return super().drift(x, t, beta=beta, use_ema=use_ema)


# ── Constants ──────────────────────────────────────────────────────────────
LOG_DIR = "lightning_logs/bs_sweep"
OFF_POLICY_RESULTS = "notebooks/offpolicy_seeds_results.json"

TOTAL_STEPS = 1000
LR = 1e-3
N_STEPS_TRAIN = 30           # SSMC random-t grid: 30 sorted points
N_STEPS_VAL = 100            # validation rollout for MAE/reward (fair across BS)
MC_SAMPLES = 10
N_TRAJ_BUFFER = 2000         # SSMC pre-gen buffer (≈ 60K samples)
N_SEEDS = 5

BS_VALUES = [1, 4, 16, 64, 256]

with open("notebooks/analytical_target.json") as f:
    E_OPT = json.load(f)["E_opt"]
all_rewards = reward_fn(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()
val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


class TrajCB(Callback):
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


@torch.no_grad()
def generate_ssmc_buffer(n_traj=N_TRAJ_BUFFER, chunk=400):
    xs, ts, ys = [], [], []
    for start in range(0, n_traj, chunk):
        b = min(chunk, n_traj - start)
        ax, at, ay = single_seed_mc(
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
        xs.append(ax.cpu()); ts.append(at.cpu()); ys.append(ay.cpu())
    x_buf = torch.cat(xs, 0); t_buf = torch.cat(ts, 0); y_buf = torch.cat(ys, 0)
    perm = torch.randperm(x_buf.shape[0])
    return x_buf[perm], t_buf[perm].unsqueeze(-1), y_buf[perm].unsqueeze(-1)


def run_one(method, bs, seed):
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
        print(f"  buffer: {tuple(x_buf.shape)}  ({(time.time() - t0)/60:.1f} min)", flush=True)
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

    tcb = TrajCB(anal_fn)
    logger = CSVLogger(LOG_DIR, name=name, version=0)
    tr = L.Trainer(
        max_steps=TOTAL_STEPS, val_check_interval=max(1, TOTAL_STEPS // 30),
        callbacks=[tcb], logger=logger,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    tr.fit(model, loader, val_dataloaders=val_loader)
    print(f"  {method} bs={bs} seed={seed} elapsed: {(time.time() - t0)/60:.1f} min", flush=True)
    del model, vm, tr, loader, ds
    if method == "ssmc":
        del x_buf, t_buf, y_buf
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── Run all (BS=256 off-policy: reuse existing) ───────────────────────────
print(f"E_OPT = {E_OPT:.4f}")
print(f"Device: {DEVICE}")
print(f"BS grid:    {BS_VALUES}")
print(f"seeds:      {N_SEEDS}")
print(f"SSMC config: smc_value=0.1·h·t, mc={MC_SAMPLES}, n_steps={N_STEPS_TRAIN} random",
      flush=True)
t_total0 = time.time()
for method in ["off", "ssmc"]:
    for bs in BS_VALUES:
        if method == "off" and bs == 256:
            print(f"\n[off bs=256] skipping — reusing first 5 of "
                  f"lightning_logs/offpolicy_seeds/.", flush=True)
            continue
        for s in range(N_SEEDS):
            run_one(method, bs, s)
            print(f"  total elapsed: {(time.time() - t_total0)/60:.1f} min", flush=True)


# ── Aggregate ─────────────────────────────────────────────────────────────
def load_curves(csv_path, max_step=TOTAL_STEPS):
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path); df = df[df["step"] <= max_step]
    return df.dropna(subset=["val_reward_mean"]), df.dropna(subset=["traj_avg_mae_guided"])


results = {}  # (method, bs) -> dict
for method in ["off", "ssmc"]:
    for bs in BS_VALUES:
        bests, maes, vcs, mcs = [], [], [], []
        for s in range(N_SEEDS):
            if method == "off" and bs == 256:
                csv = f"lightning_logs/offpolicy_seeds/seed_{s:02d}/version_0/metrics.csv"
            else:
                csv = f"{LOG_DIR}/{method}_bs{bs:03d}_seed{s:02d}/version_0/metrics.csv"
            v, m = load_curves(csv)
            if v is None or len(v) == 0:
                continue
            bests.append(float(v["val_reward_mean"].max()))
            maes.append(float(m["traj_avg_mae_guided"].iloc[-1]) if len(m) else float("nan"))
            vcs.append(v); mcs.append(m)
        results[(method, bs)] = dict(
            bests=np.array(bests), maes=np.array(maes),
            val_curves=vcs, mae_curves=mcs,
        )


print("\n" + "=" * 78)
print(f"{'method':>6} {'bs':>5} {'n':>3}  {'best mean':>10}  {'best sd':>8}  "
      f"{'MAE mean':>10}  {'MAE sd':>8}")
print("=" * 78)
for method in ["off", "ssmc"]:
    for bs in BS_VALUES:
        r = results[(method, bs)]
        if len(r["bests"]) == 0:
            continue
        print(f"{method:>6} {bs:>5} {len(r['bests']):>3}  "
              f"{r['bests'].mean():>10.3f}  "
              f"{r['bests'].std(ddof=1) if len(r['bests'])>1 else 0:>8.3f}  "
              f"{r['maes'].mean():>10.3f}  "
              f"{r['maes'].std(ddof=1) if len(r['maes'])>1 else 0:>8.3f}")
print("=" * 78)


# ── Plot: 2 rows × 5 cols ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, len(BS_VALUES), figsize=(4 * len(BS_VALUES), 8), sharex=True)
fig.suptitle(
    f"Batch-size sweep — off-policy vs SSMC  "
    f"(smc_value = 0.1·h(x)·t, mc={MC_SAMPLES}, n_steps={N_STEPS_TRAIN} random)  "
    f"— {N_SEEDS} seeds per cell",
    fontsize=11, fontweight="bold",
)
OFF_C = "#1f77b4"; SSMC_C = "#ff7f0e"

for col, bs in enumerate(BS_VALUES):
    # reward
    ax = axes[0, col]
    ax.set_title(f"BS = {bs}")
    for v in results[("off", bs)]["val_curves"]:
        ax.plot(v["step"], v["val_reward_mean"], color=OFF_C, alpha=0.55, lw=1.2)
    for v in results[("ssmc", bs)]["val_curves"]:
        ax.plot(v["step"], v["val_reward_mean"], color=SSMC_C, alpha=0.55, lw=1.2)
    ax.axhline(E_OPT, color="black", ls=":", alpha=0.6, label=f"E_opt={E_OPT:.2f}")
    ax.plot([], [], color=OFF_C, lw=2, label="off-policy")
    ax.plot([], [], color=SSMC_C, lw=2, label="SSMC")
    if col == 0:
        ax.set_ylabel("avg terminal reward")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="lower right")

    # MAE
    ax = axes[1, col]
    for m in results[("off", bs)]["mae_curves"]:
        if len(m) > 0:
            ax.plot(m["step"], m["traj_avg_mae_guided"], color=OFF_C, alpha=0.55, lw=1.2)
    for m in results[("ssmc", bs)]["mae_curves"]:
        if len(m) > 0:
            ax.plot(m["step"], m["traj_avg_mae_guided"], color=SSMC_C, alpha=0.55, lw=1.2)
    ax.plot([], [], color=OFF_C, lw=2, label="off-policy")
    ax.plot([], [], color=SSMC_C, lw=2, label="SSMC")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    if col == 0:
        ax.set_ylabel("MAE  |V_θ − V*|")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
out = "notebooks/bs_sweep.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")


# ── Save JSON ─────────────────────────────────────────────────────────────
summary = {
    "config": {
        "TOTAL_STEPS": TOTAL_STEPS, "LR": LR,
        "N_STEPS_TRAIN": N_STEPS_TRAIN, "N_STEPS_VAL": N_STEPS_VAL,
        "MC_SAMPLES": MC_SAMPLES, "N_TRAJ_BUFFER": N_TRAJ_BUFFER,
        "include_t_zero": False, "random_t": True,
        "smc_value": "0.1 * h(x) * t",
        "BS_values": BS_VALUES, "N_SEEDS": N_SEEDS,
        "E_OPT": E_OPT,
    },
    "by_cell": {
        f"{method}_bs={bs}": {
            "best_reward_values": results[(method, bs)]["bests"].tolist(),
            "final_guided_mae_values": results[(method, bs)]["maes"].tolist(),
        }
        for method in ["off", "ssmc"] for bs in BS_VALUES
    },
}
with open("notebooks/bs_sweep_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nTotal elapsed: {(time.time() - t_total0)/60:.1f} min")
print("Done.")
