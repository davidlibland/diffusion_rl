#!/usr/bin/env python3
"""Sweep over k where smc_value(x, t) = k * value(x, t) (live value module).

Uses OnPolicySMCDataset so the twist tracks the trained value function each
regen.  Crossed with off_policy_frac in {0, 0.1, 0.3}.  No-EMA drift,
include_t_zero=False, 1 seed per (k, off_frac) cell.
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

from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
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


class OnPolicyValueLive(OnPolicyValue):
    def drift(self, x, t, beta=1, use_ema=False):
        return super().drift(x, t, beta=beta, use_ema=use_ema)


# ── Constants ──────────────────────────────────────────────────────────────
LOG_DIR = "lightning_logs/ssmc_k_value_sweep"
OFF_POLICY_RESULTS = "experiments/misc/2026-05-28_offpolicy_seeds/offpolicy_seeds_results.json"

TOTAL_STEPS = 1000
BS = 256
LR = 1e-3
N_STEPS = 100
MC_SAMPLES = 10
DS_BATCH = 8                  # larger than prior to reduce regen overhead
N_SEEDS = 1

K_VALUES = [0.1, 0.3, 1.0]      # geometric grid 0.1 → 1.0 (ratio ≈ 3.16)
OFF_FRACS = [0.0, 0.1, 0.3]

with open("experiments/common/analytical_target.json") as f:
    E_OPT = json.load(f)["E_opt"]

all_rewards = reward_fn(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()
val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


class TrajCB(Callback):
    def __init__(self, af, n=256, ns=N_STEPS):
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


def run_one(k, off_frac, seed):
    name = f"k{k:.4f}_f{off_frac:.2f}_seed{seed:02d}"
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
    print(f"\n=== k={k}  off_frac={off_frac}  seed={seed} ===", flush=True)
    t0 = time.time()

    vm = ValueNetwork(D, bias=bias_val)
    model = OnPolicyValueLive(
        base_score_module=base_drift, value_module=vm, reward_function=reward_fn,
        dim=D, a=a, lr=LR, loss_type="quad",
        analytical_value_fn=anal_fn, ema_decay=0.999,
    )

    # Live twist: smc_value(x, t) = k * value(x, t).
    def smc_value_k(x, t):
        v = model.value_module(x, t)
        return k * v.reshape(-1)

    ds = OnPolicySMCDataset(
        dim=D,
        drift=base_drift,
        value=model.value_module,
        smc_value=smc_value_k,
        reward=reward_fn,
        device=DEVICE,
        a=a,
        batch_size=DS_BATCH,
        n_steps=N_STEPS,
        mc_samples_per_step=MC_SAMPLES,
        sampling_method="single_seed_mc",
        off_policy_frac=off_frac,
        include_t_zero=False,
        generating_function=gmm_sample,
    )
    loader = DataLoader(ds, batch_size=BS)
    tcb = TrajCB(anal_fn)
    logger = CSVLogger(LOG_DIR, name=name, version=0)
    tr = L.Trainer(
        max_steps=TOTAL_STEPS, val_check_interval=max(1, TOTAL_STEPS // 30),
        callbacks=[tcb], logger=logger,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    tr.fit(model, loader, val_dataloaders=val_loader)
    print(f"  elapsed: {(time.time() - t0)/60:.1f} min", flush=True)
    del model, vm, tr, loader, ds
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── Run all ────────────────────────────────────────────────────────────────
print(f"E_OPT = {E_OPT:.4f}")
print(f"Device: {DEVICE}")
print(f"k grid: {K_VALUES}")
print(f"off_frac grid: {OFF_FRACS}")
print(f"seeds per cell: {N_SEEDS}\n", flush=True)
t_total0 = time.time()
for k in K_VALUES:
    for off_frac in OFF_FRACS:
        for s in range(N_SEEDS):
            run_one(k, off_frac, s)
            print(f"  total elapsed: {(time.time() - t_total0)/60:.1f} min", flush=True)


# ── Aggregate ─────────────────────────────────────────────────────────────
def load_curves(csv_path, max_step=TOTAL_STEPS):
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path); df = df[df["step"] <= max_step]
    return df.dropna(subset=["val_reward_mean"]), df.dropna(subset=["traj_avg_mae_guided"])


results = {}  # (k, off_frac) -> dict
for k in K_VALUES:
    for off_frac in OFF_FRACS:
        bests, maes, vcs, mcs = [], [], [], []
        for s in range(N_SEEDS):
            name = f"k{k:.4f}_f{off_frac:.2f}_seed{s:02d}"
            v, m = load_curves(f"{LOG_DIR}/{name}/version_0/metrics.csv")
            if v is None or len(v) == 0:
                continue
            bests.append(float(v["val_reward_mean"].max()))
            maes.append(float(m["traj_avg_mae_guided"].iloc[-1]) if len(m) else float("nan"))
            vcs.append(v); mcs.append(m)
        results[(k, off_frac)] = dict(
            bests=np.array(bests), maes=np.array(maes),
            val_curves=vcs, mae_curves=mcs,
        )

with open(OFF_POLICY_RESULTS) as f:
    off = json.load(f)
off_bests = np.array(off["best_reward"]["values"])
off_maes = np.array(off["final_guided_mae"]["values"])
off_vcs, off_mcs = [], []
for s in range(off["n_seeds"]):
    v, m = load_curves(f"lightning_logs/offpolicy_seeds/seed_{s:02d}/version_0/metrics.csv")
    if v is not None:
        off_vcs.append(v); off_mcs.append(m)


print("\n" + "=" * 78)
print(f"{'k':>6} {'off_frac':>9}  {'best':>10}  {'final MAE':>10}")
print("=" * 78)
for k in K_VALUES:
    for f_ in OFF_FRACS:
        r = results[(k, f_)]
        if len(r["bests"]) == 0:
            continue
        print(f"{k:>6.3f} {f_:>9.2f}  {r['bests'].mean():>10.3f}  {r['maes'].mean():>10.3f}")
print(f"{'off-pol':>6} {'(n=10)':>9}  {off_bests.mean():>10.3f}  {off_maes.mean():>10.3f}")
print("=" * 78)


# ── Plot: 2×3 grid (rows = reward/MAE, cols = off_frac), k as color ───────
fig, axes = plt.subplots(2, len(OFF_FRACS), figsize=(5 * len(OFF_FRACS), 9), sharex=True)
fig.suptitle(
    f"SSMC k-sweep, smc_value=k·V(x,t),  mc={MC_SAMPLES}, no-EMA, "
    f"include_t_zero=False  (n={N_SEEDS} / cell vs off-policy n={len(off_bests)})",
    fontsize=11, fontweight="bold",
)
OFF_C = "#888888"
cmap = plt.cm.viridis
colors = {k: cmap(i / max(1, len(K_VALUES) - 1)) for i, k in enumerate(K_VALUES)}

for col, f_ in enumerate(OFF_FRACS):
    # reward
    ax = axes[0, col]
    ax.set_title(f"Reward  (off_frac = {f_})")
    for v in off_vcs:
        ax.plot(v["step"], v["val_reward_mean"], color=OFF_C, alpha=0.30, lw=1.0)
    for k in K_VALUES:
        for v in results[(k, f_)]["val_curves"]:
            ax.plot(v["step"], v["val_reward_mean"], color=colors[k], alpha=0.9, lw=1.6)
    ax.axhline(E_OPT, color="black", ls=":", alpha=0.6, label=f"E_opt={E_OPT:.2f}")
    ax.plot([], [], color=OFF_C, lw=2, label=f"off-policy (n={len(off_vcs)})")
    for k in K_VALUES:
        ax.plot([], [], color=colors[k], lw=2, label=f"k={k}")
    ax.set_ylabel("avg terminal reward")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="lower right")

    # MAE
    ax = axes[1, col]
    ax.set_title(f"Guided value MAE  (off_frac = {f_})")
    for m in off_mcs:
        if len(m) > 0:
            ax.plot(m["step"], m["traj_avg_mae_guided"], color=OFF_C, alpha=0.30, lw=1.0)
    for k in K_VALUES:
        for m in results[(k, f_)]["mae_curves"]:
            if len(m) > 0:
                ax.plot(m["step"], m["traj_avg_mae_guided"], color=colors[k], alpha=0.9, lw=1.6)
    ax.plot([], [], color=OFF_C, lw=2, label=f"off-policy (n={len(off_mcs)})")
    for k in K_VALUES:
        ax.plot([], [], color=colors[k], lw=2, label=f"k={k}")
    ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel("MAE  |V_θ − V*|")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
out = "experiments/misc/2026-05-28_ssmc_sweeps/ssmc_k_value_sweep.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")


# ── Save JSON ─────────────────────────────────────────────────────────────
summary = {
    "config": {
        "TOTAL_STEPS": TOTAL_STEPS, "BS": BS, "LR": LR,
        "N_STEPS": N_STEPS, "MC_SAMPLES": MC_SAMPLES, "DS_BATCH": DS_BATCH,
        "N_SEEDS": N_SEEDS,
        "include_t_zero": False, "use_ema": False,
        "smc_value": "k * value(x, t)",
        "k_values": K_VALUES, "off_fracs": OFF_FRACS, "E_OPT": E_OPT,
    },
    "by_cell": {
        f"k={k}_f={f_}": {
            "best_reward_values": results[(k, f_)]["bests"].tolist(),
            "final_guided_mae_values": results[(k, f_)]["maes"].tolist(),
        }
        for k in K_VALUES for f_ in OFF_FRACS
    },
    "off_policy_baseline": {
        "best_reward_mean": float(off_bests.mean()),
        "best_reward_sd": float(off_bests.std(ddof=1)),
        "final_guided_mae_mean": float(off_maes.mean()),
        "final_guided_mae_sd": float(off_maes.std(ddof=1)),
        "n_seeds": len(off_bests),
    },
}
with open("experiments/misc/2026-05-28_ssmc_sweeps/ssmc_k_value_sweep_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nTotal elapsed: {(time.time() - t_total0)/60:.1f} min")
print("Done.")
