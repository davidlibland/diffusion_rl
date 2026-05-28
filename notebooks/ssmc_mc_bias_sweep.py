#!/usr/bin/env python3
"""How does SSMC value-estimate bias evolve with mc_samples?

Finite mc_samples → biased per-step importance weights and biased log_z
ratios; single_seed_mc telescopes that bias over all n_steps.  This sweep
measures the *signed* bias as mc_samples grows, at small batch sizes.

SSMC only.  smc_value = 0.1·h(x)·t, n_steps=30 random, no-EMA,
include_t_zero=False.

mc_samples ∈ {1, 3, 10, 25, 50},  BS ∈ {1, 4},  3 seeds.

Bias metrics (signed; positive = V overestimated):
  v0_bias        = val_value_at_t0 − V_0_0      (canonical point bias)
  off_dist_bias  = mean(V_pred − V_anal) on a fixed Brownian-bridge eval set
  ssmc_dist_bias = mean(V_pred − V_anal) on a fixed mc=50 SSMC eval set
plus the corresponding MAEs and best val_reward.
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
LOG_DIR = "lightning_logs/ssmc_mc_bias_sweep"

TOTAL_STEPS = 1000
LR = 1e-3
N_STEPS_TRAIN = 30
N_STEPS_VAL = 100
N_TRAJ_BUFFER = 2000
N_EVAL = 2048
N_SEEDS = 3

MC_VALUES = [1, 3, 10, 25, 50]
BS_VALUES = [1, 4]
EVAL_REF_MC = 50             # fixed mc for the SSMC eval set (constant coverage)

with open("notebooks/analytical_target.json") as f:
    _at = json.load(f)
E_OPT = _at["E_opt"]
V_0_0 = _at["V_0_0"]         # true V(0,0); val_value_at_t0 should match this

all_rewards = reward_fn(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()
val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


# ── Eval-set builders ─────────────────────────────────────────────────────
@torch.no_grad()
def make_off_dist_eval(n=N_EVAL):
    x1 = torch.from_numpy(gmm_sample(n)).float()
    t = torch.rand(n)
    eps = torch.randn(n, D)
    x = t[:, None] * x1 + torch.sqrt(2 * a * t * (1 - t))[:, None] * eps
    return x, t


@torch.no_grad()
def make_ssmc_dist_eval(mc, n=N_EVAL, chunk=400):
    n_traj = (n + N_STEPS_TRAIN - 1) // N_STEPS_TRAIN
    xs, ts = [], []
    for start in range(0, n_traj, chunk):
        b = min(chunk, n_traj - start)
        ax, at, _ = single_seed_mc(
            drift=base_drift, value=zero_value, log_tau=smc_value_fn, h=reward_fn,
            a=a, batch_size=b, mc_samples=mc, dim=D, n_steps=N_STEPS_TRAIN,
            device=DEVICE, random_t=True, include_t_zero=False,
        )
        xs.append(ax.cpu()); ts.append(at.cpu())
    x_all = torch.cat(xs, 0); t_all = torch.cat(ts, 0)
    idx = torch.randperm(x_all.shape[0])[:n]
    return x_all[idx], t_all[idx]


N_TBINS = 10  # per-t bias resolution


class BalancedMAECallback(Callback):
    def __init__(self, eval_x, eval_t, af, name: str, n_tbins: int = N_TBINS):
        super().__init__()
        self.eval_x = eval_x.detach(); self.eval_t = eval_t.detach()
        self.af = af; self.name = name
        self.n_tbins = n_tbins
        self.t_edges = torch.linspace(0, 1, n_tbins + 1)

    def on_validation_batch_end(self, trainer, pl, outputs, batch, bi):
        if bi > 0:
            return
        x = self.eval_x.to(pl.device); t = self.eval_t.to(pl.device)
        with torch.no_grad():
            v_pred = pl.value_module(x, t)
            v_anal = self.af(x, t)
        err = v_pred - v_anal
        pl.log(f"{self.name}_mae", err.abs().mean(), prog_bar=False)
        pl.log(f"{self.name}_bias", err.mean(), prog_bar=False)
        # per-t-bin signed bias (tests telescoping hypothesis: |bias| ↑ as t→0)
        for i in range(self.n_tbins):
            lo = float(self.t_edges[i]); hi = float(self.t_edges[i + 1])
            if i < self.n_tbins - 1:
                mask = (t >= lo) & (t < hi)
            else:
                mask = (t >= lo) & (t <= hi)
            if mask.sum() > 0:
                pl.log(f"{self.name}_bias_tb{i:02d}", err[mask].mean(),
                       prog_bar=False)


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
def generate_ssmc_buffer(mc, n_traj=N_TRAJ_BUFFER, chunk=400):
    xs, ts, ys = [], [], []
    for start in range(0, n_traj, chunk):
        b = min(chunk, n_traj - start)
        ax, at, ay = single_seed_mc(
            drift=base_drift, value=zero_value, log_tau=smc_value_fn, h=reward_fn,
            a=a, batch_size=b, mc_samples=mc, dim=D, n_steps=N_STEPS_TRAIN,
            device=DEVICE, random_t=True, include_t_zero=False,
        )
        xs.append(ax.cpu()); ts.append(at.cpu()); ys.append(ay.cpu())
    x_buf = torch.cat(xs, 0); t_buf = torch.cat(ts, 0); y_buf = torch.cat(ys, 0)
    perm = torch.randperm(x_buf.shape[0])
    return x_buf[perm], t_buf[perm].unsqueeze(-1), y_buf[perm].unsqueeze(-1)


def run_one(mc, bs, seed, eval_off, eval_ssmc):
    name = f"mc{mc:03d}_bs{bs:03d}_seed{seed:02d}"
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
    print(f"\n=== mc={mc}  bs={bs}  seed={seed} ===", flush=True)
    t0 = time.time()

    vm = ValueNetwork(D, bias=bias_val)
    model = OnPolicyValueLive(
        base_score_module=base_drift, value_module=vm, reward_function=reward_fn,
        dim=D, a=a, lr=LR, loss_type="quad",
        analytical_value_fn=anal_fn, ema_decay=0.999,
    )
    x_buf, t_buf, y_buf = generate_ssmc_buffer(mc)
    ds = TensorDataset(y_buf, x_buf, t_buf)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    cbs = [
        TrajCB(anal_fn),
        BalancedMAECallback(eval_off[0], eval_off[1], anal_fn, name="off_dist"),
        BalancedMAECallback(eval_ssmc[0], eval_ssmc[1], anal_fn, name="ssmc_dist"),
    ]
    logger = CSVLogger(LOG_DIR, name=name, version=0)
    tr = L.Trainer(
        max_steps=TOTAL_STEPS, val_check_interval=max(1, TOTAL_STEPS // 30),
        callbacks=cbs, logger=logger,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    tr.fit(model, loader, val_dataloaders=val_loader)
    print(f"  mc={mc} bs={bs} seed={seed} elapsed: {(time.time() - t0)/60:.1f} min",
          flush=True)
    del model, vm, tr, loader, ds, x_buf, t_buf, y_buf
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── Fixed eval sets ───────────────────────────────────────────────────────
print("Building fixed eval sets...", flush=True)
torch.manual_seed(12345); np.random.seed(12345)
eval_off = make_off_dist_eval()
eval_ssmc = make_ssmc_dist_eval(EVAL_REF_MC)
print(f"  off_dist eval x {tuple(eval_off[0].shape)} | "
      f"ssmc_dist(mc={EVAL_REF_MC}) eval x {tuple(eval_ssmc[0].shape)}", flush=True)

print(f"\nE_OPT = {E_OPT:.4f}   V_0_0 = {V_0_0:.4f}")
print(f"Device: {DEVICE}")
print(f"mc grid: {MC_VALUES}   BS: {BS_VALUES}   seeds: {N_SEEDS}", flush=True)
t_total0 = time.time()
for mc in MC_VALUES:
    for bs in BS_VALUES:
        for s in range(N_SEEDS):
            run_one(mc, bs, s, eval_off, eval_ssmc)
            print(f"  total elapsed: {(time.time() - t_total0)/60:.1f} min", flush=True)


# ── Aggregate at best-val-reward step ─────────────────────────────────────
def best_step_metrics(csv_path, metrics, monitor="val_reward_mean"):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    v = df.dropna(subset=[monitor])
    if len(v) == 0:
        return None
    best_step = int(v.loc[v[monitor].idxmax(), "step"])
    out = {"best_step": best_step, monitor: float(v[monitor].max())}
    for m in metrics:
        sub = df.dropna(subset=[m]) if m in df.columns else df.iloc[0:0]
        if len(sub) == 0:
            out[m] = float("nan"); continue
        sub2 = sub[sub["step"] <= best_step]
        if len(sub2) == 0:
            sub2 = sub
        out[m] = float(sub2.iloc[-1][m])
    return out


METRICS = ["val_value_at_t0",
           "off_dist_mae", "off_dist_bias",
           "ssmc_dist_mae", "ssmc_dist_bias",
           "traj_avg_mae_guided"]
agg = {}
for mc in MC_VALUES:
    for bs in BS_VALUES:
        cells = []
        for s in range(N_SEEDS):
            csv = f"{LOG_DIR}/mc{mc:03d}_bs{bs:03d}_seed{s:02d}/version_0/metrics.csv"
            r = best_step_metrics(csv, METRICS)
            if r is not None:
                # signed bias of V(0,0) vs the true value
                r["v0_bias"] = r["val_value_at_t0"] - V_0_0
                cells.append(r)
        agg[(mc, bs)] = cells


def mean_sd(vals):
    arr = np.array(vals, dtype=float); arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0)


print("\n" + "=" * 110)
print(f"{'mc':>4} {'BS':>3} {'n':>2} | {'best_R':>11} | "
      f"{'v0_bias':>13} | {'off_bias':>13} | {'ssmc_bias':>13} | "
      f"{'off_mae':>9} | {'ssmc_mae':>9}")
print("=" * 110)
for mc in MC_VALUES:
    for bs in BS_VALUES:
        cells = agg[(mc, bs)]
        if not cells:
            continue
        mr, sr = mean_sd([x["val_reward_mean"] for x in cells])
        mv, sv = mean_sd([x["v0_bias"] for x in cells])
        mo, so = mean_sd([x["off_dist_bias"] for x in cells])
        ms, ss = mean_sd([x["ssmc_dist_bias"] for x in cells])
        moa, _ = mean_sd([x["off_dist_mae"] for x in cells])
        msa, _ = mean_sd([x["ssmc_dist_mae"] for x in cells])
        print(f"{mc:>4} {bs:>3} {len(cells):>2} | "
              f"{mr:>6.2f}±{sr:>4.2f} | "
              f"{mv:>+7.3f}±{sv:>4.2f} | "
              f"{mo:>+7.3f}±{so:>4.2f} | "
              f"{ms:>+7.3f}±{ss:>4.2f} | "
              f"{moa:>9.3f} | {msa:>9.3f}")
print("=" * 110)
print("bias sign: + = V overestimated,  − = V underestimated")


# ── Plot: signed bias vs mc_samples ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle(
    "SSMC value-estimate bias vs mc_samples  (smc_value=0.1·h·t, "
    f"n_steps={N_STEPS_TRAIN} random, no-EMA, {N_SEEDS} seeds)",
    fontsize=12, fontweight="bold",
)
BS_C = {1: "#d62728", 4: "#1f77b4"}

def series(metric, bs):
    xs, ms, ss = [], [], []
    for mc in MC_VALUES:
        cells = agg[(mc, bs)]
        if not cells:
            continue
        m, s = mean_sd([x[metric] for x in cells])
        xs.append(mc); ms.append(m); ss.append(s)
    return np.array(xs), np.array(ms), np.array(ss)

for ax, metric, title in [
    (axes[0], "v0_bias", "V(0,0) bias  (val_value_at_t0 − V_0_0)"),
    (axes[1], "off_dist_bias", "Bias on off-policy eval dist"),
    (axes[2], "ssmc_dist_bias", f"Bias on SSMC eval dist (mc={EVAL_REF_MC})"),
]:
    for bs in BS_VALUES:
        xs, ms, ss = series(metric, bs)
        ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=4,
                    color=BS_C[bs], label=f"BS={bs}")
    ax.axhline(0, color="black", ls=":", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xticks(MC_VALUES); ax.set_xticklabels(MC_VALUES)
    ax.set_xlabel("mc_samples")
    ax.set_ylabel("signed bias  (V_pred − V_true)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=9)

plt.tight_layout()
out = "notebooks/ssmc_mc_bias_sweep.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nSaved: {out}")


# ── Bias vs t (per-t-bin, tests telescoping hypothesis) ───────────────────
TB_METRICS = [f"ssmc_dist_bias_tb{i:02d}" for i in range(N_TBINS)]
OFFTB_METRICS = [f"off_dist_bias_tb{i:02d}" for i in range(N_TBINS)]
tb_centers = (np.linspace(0, 1, N_TBINS + 1)[:-1]
              + np.linspace(0, 1, N_TBINS + 1)[1:]) / 2

agg_tb = {}
for mc in MC_VALUES:
    for bs in BS_VALUES:
        cells = []
        for s in range(N_SEEDS):
            csv = f"{LOG_DIR}/mc{mc:03d}_bs{bs:03d}_seed{s:02d}/version_0/metrics.csv"
            r = best_step_metrics(csv, TB_METRICS + OFFTB_METRICS)
            if r is not None:
                cells.append(r)
        agg_tb[(mc, bs)] = cells

fig, axes = plt.subplots(2, len(BS_VALUES), figsize=(7 * len(BS_VALUES), 10),
                         sharex=True)
fig.suptitle(
    "SSMC signed bias vs t  (hypothesis: |bias| grows as t→0, →0 as t→1)\n"
    f"smc_value=0.1·h·t, n_steps={N_STEPS_TRAIN} random, no-EMA, "
    f"{N_SEEDS} seeds, @best-reward ckpt",
    fontsize=12, fontweight="bold",
)
cmap = plt.cm.viridis
mc_colors = {mc: cmap(i / max(1, len(MC_VALUES) - 1))
             for i, mc in enumerate(MC_VALUES)}

for col, bs in enumerate(BS_VALUES):
    for row, (mlist, dist_label) in enumerate([
        (TB_METRICS, f"SSMC eval dist (mc={EVAL_REF_MC})"),
        (OFFTB_METRICS, "off-policy eval dist"),
    ]):
        ax = axes[row, col]
        ax.set_title(f"BS = {bs}  ·  {dist_label}")
        for mc in MC_VALUES:
            cells = agg_tb[(mc, bs)]
            if not cells:
                continue
            curve_m, curve_s = [], []
            for m in mlist:
                vals = np.array([cc.get(m, np.nan) for cc in cells], dtype=float)
                vals = vals[np.isfinite(vals)]
                curve_m.append(vals.mean() if len(vals) else np.nan)
                curve_s.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
            ax.errorbar(tb_centers, curve_m, yerr=curve_s, marker="o",
                        ms=4, capsize=3, color=mc_colors[mc], label=f"mc={mc}")
        ax.axhline(0, color="black", ls=":", alpha=0.7)
        ax.set_xlabel("t (bin center)")
        ax.set_ylabel("signed bias  (V_pred − V_true)")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8, title="mc_samples")

plt.tight_layout()
out_t = "notebooks/ssmc_mc_bias_vs_t.png"
plt.savefig(out_t, dpi=140, bbox_inches="tight")
print(f"Saved: {out_t}")

with open("notebooks/ssmc_mc_bias_sweep_results.json", "w") as f:
    json.dump({
        "config": {
            "TOTAL_STEPS": TOTAL_STEPS, "LR": LR,
            "N_STEPS_TRAIN": N_STEPS_TRAIN, "N_TRAJ_BUFFER": N_TRAJ_BUFFER,
            "N_EVAL": N_EVAL, "EVAL_REF_MC": EVAL_REF_MC,
            "MC_VALUES": MC_VALUES, "BS_VALUES": BS_VALUES, "N_SEEDS": N_SEEDS,
            "smc_value": "0.1 * h(x) * t", "E_OPT": E_OPT, "V_0_0": V_0_0,
        },
        "by_cell": {
            f"mc={mc}_bs={bs}": agg[(mc, bs)]
            for mc in MC_VALUES for bs in BS_VALUES
        },
    }, f, indent=2)

print(f"\nTotal elapsed: {(time.time() - t_total0)/60:.1f} min")
print("Done.")
