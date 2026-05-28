#!/usr/bin/env python3
"""5 seeds of SSMC k=0 with fully-shuffled buffer (full trajectory length).

Mirrors the setup of `ssmc_shuffled_buffer.py`:
  - 2000 trajectories per seed × 100 steps = 200K samples (correlated within
    each trajectory, but globally shuffled before training).
  - log_tau(x, t) = 0  (no SMC twisting; just base-SDE forward).
  - Each batch of 256 then samples from up to 256 distinct trajectories.

After all 5 seeds finish, this script merges off-policy seeds results from
`notebooks/offpolicy_seeds_results.json` and produces a side-by-side plot
of the two distributions.
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


# ── Constants ──────────────────────────────────────────────────────────────
LOG_DIR = "lightning_logs/ssmc_shuffle_seeds"
CKPT_DIR = "checkpoints/ssmc_shuffle_seeds"
OFF_POLICY_RESULTS = "notebooks/offpolicy_seeds_results.json"

TOTAL_STEPS = 1000
BS = 256
LR = 1e-3
N_STEPS = 100
MC_SAMPLES = 10
N_TRAJ_BIG = 2000        # 2000 × 100 = 200K samples per buffer
N_SEEDS = 5

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


@torch.no_grad()
def generate_buffer(n_traj, chunk=400):
    """Generate `n_traj` SSMC trajectories with log_tau=0; returns shuffled buffers."""
    print(f"  Generating {n_traj} trajectories on {DEVICE}...")
    t0 = time.time()
    xs, ts, ys = [], [], []
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
        xs.append(all_x.cpu())
        ts.append(all_t.cpu())
        ys.append(all_tgt.cpu())
    x_buf = torch.cat(xs, dim=0)
    t_buf = torch.cat(ts, dim=0)
    y_buf = torch.cat(ys, dim=0)
    perm = torch.randperm(x_buf.shape[0])
    x_buf = x_buf[perm]
    t_buf = t_buf[perm].unsqueeze(-1)
    y_buf = y_buf[perm].unsqueeze(-1)
    print(f"    buffer: {tuple(x_buf.shape)}  ({(time.time() - t0)/60:.1f} min)")
    return x_buf, t_buf, y_buf


def run_one(seed):
    name = f"seed_{seed:02d}"
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

    L.seed_everything(seed, workers=True)
    print(f"\n=== seed = {seed} ===")
    x_buf, t_buf, y_buf = generate_buffer(N_TRAJ_BIG)

    print(f"  [training] {name}")
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
    del model, vm, trainer, loader, ds, x_buf, t_buf, y_buf
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── Run all seeds ──────────────────────────────────────────────────────────
print(f"E_OPT = {E_OPT:.4f}")
print(f"Device: {DEVICE}")
print(f"BS={BS}  LR={LR}  TOTAL_STEPS={TOTAL_STEPS}  N_TRAJ_BIG={N_TRAJ_BIG}")
print(f"seeds = 0..{N_SEEDS - 1}\n")

t_total0 = time.time()
for s in range(N_SEEDS):
    run_one(s)
    print(f"  total elapsed so far: {(time.time() - t_total0)/60:.1f} min")


# ── Aggregate and plot vs off-policy distribution ────────────────────────
def load_curves(csv_path, max_step=TOTAL_STEPS):
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    df = df[df["step"] <= max_step]
    val = df.dropna(subset=["val_reward_mean"])
    mae = df.dropna(subset=["traj_avg_mae_guided"])
    return val, mae


ssmc_bests, ssmc_finals, ssmc_maes = [], [], []
ssmc_val_curves, ssmc_mae_curves = [], []
for s in range(N_SEEDS):
    val, mae = load_curves(f"{LOG_DIR}/seed_{s:02d}/version_0/metrics.csv")
    if val is None or len(val) == 0:
        continue
    ssmc_bests.append(float(val["val_reward_mean"].max()))
    ssmc_finals.append(float(val["val_reward_mean"].iloc[-1]))
    ssmc_maes.append(float(mae["traj_avg_mae_guided"].iloc[-1]) if len(mae) else float("nan"))
    ssmc_val_curves.append(val)
    ssmc_mae_curves.append(mae)
ssmc_bests = np.array(ssmc_bests)
ssmc_finals = np.array(ssmc_finals)
ssmc_maes = np.array(ssmc_maes)

# Off-policy seeds (already run).
off_data = None
off_val_curves, off_mae_curves = [], []
if os.path.exists(OFF_POLICY_RESULTS):
    with open(OFF_POLICY_RESULTS) as f:
        off_data = json.load(f)
    for s in range(off_data["n_seeds"]):
        val, mae = load_curves(f"lightning_logs/offpolicy_seeds/seed_{s:02d}/version_0/metrics.csv")
        if val is not None:
            off_val_curves.append(val)
            off_mae_curves.append(mae)

print(f"\n{'=' * 72}")
print(f"  SSMC k=0 SHUFFLED ({len(ssmc_bests)} seeds, N_TRAJ={N_TRAJ_BIG})")
print(f"{'=' * 72}")
print(f"  best reward      mean={ssmc_bests.mean():>7.3f}   sd={ssmc_bests.std(ddof=1):>5.3f}   "
      f"min={ssmc_bests.min():>7.3f}  max={ssmc_bests.max():>7.3f}")
print(f"  final reward     mean={ssmc_finals.mean():>7.3f}   sd={ssmc_finals.std(ddof=1):>5.3f}")
print(f"  final guided MAE mean={ssmc_maes.mean():>6.3f}   sd={ssmc_maes.std(ddof=1):>5.3f}   "
      f"min={ssmc_maes.min():>6.3f}  max={ssmc_maes.max():>6.3f}")

if off_data is not None:
    off_bests = np.array(off_data["best_reward"]["values"])
    off_maes = np.array(off_data["final_guided_mae"]["values"])
    off_finals = np.array(off_data["final_reward"]["values"])
    print(f"\n  OFF-POLICY ({off_data['n_seeds']} seeds, for reference)")
    print(f"  best reward      mean={off_bests.mean():>7.3f}   sd={off_bests.std(ddof=1):>5.3f}")
    print(f"  final guided MAE mean={off_maes.mean():>6.3f}   sd={off_maes.std(ddof=1):>5.3f}")

    # Welch t-tests on best reward and final MAE
    from scipy import stats
    t_best, p_best = stats.ttest_ind(ssmc_bests, off_bests, equal_var=False)
    t_mae, p_mae = stats.ttest_ind(ssmc_maes, off_maes, equal_var=False)
    print(f"\n  Welch t-test  (off-policy vs SSMC)")
    print(f"    best reward:      t={t_best:+.2f}  p={p_best:.3f}")
    print(f"    final guided MAE: t={t_mae:+.2f}  p={p_mae:.3f}")


# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    f"Off-policy ({len(off_val_curves)} seeds) vs SSMC k=0 shuffled ({len(ssmc_val_curves)} seeds)  —  1000 steps",
    fontsize=12, fontweight="bold",
)

# (a) Reward curves
ax = axes[0, 0]
ax.set_title("Reward curves per seed")
for val in off_val_curves:
    ax.plot(val["step"], val["val_reward_mean"], color="#1f77b4", alpha=0.4, lw=1.0)
for val in ssmc_val_curves:
    ax.plot(val["step"], val["val_reward_mean"], color="#d62728", alpha=0.5, lw=1.2)
ax.axhline(E_OPT, color="black", ls=":", alpha=0.6, label=f"E_opt={E_OPT:.2f}")
ax.plot([], [], color="#1f77b4", lw=2, label="off-policy seeds")
ax.plot([], [], color="#d62728", lw=2, label="ssmc k=0 shuffled seeds")
ax.set_xlabel("training step")
ax.set_ylabel("avg terminal reward")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (b) Histogram of best reward
ax = axes[0, 1]
ax.set_title("Best reward distribution")
if off_data is not None:
    ax.hist(off_bests, bins=8, color="#1f77b4", alpha=0.6, edgecolor="white",
            label=f"off-policy n={len(off_bests)}")
ax.hist(ssmc_bests, bins=5, color="#d62728", alpha=0.6, edgecolor="white",
        label=f"ssmc k=0 shuf n={len(ssmc_bests)}")
ax.set_xlabel("best val_reward_mean")
ax.set_ylabel("count")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) MAE curves
ax = axes[1, 0]
ax.set_title("Guided-trajectory value MAE per seed")
for mae in off_mae_curves:
    if len(mae) > 0:
        ax.plot(mae["step"], mae["traj_avg_mae_guided"], color="#1f77b4", alpha=0.4, lw=1.0)
for mae in ssmc_mae_curves:
    if len(mae) > 0:
        ax.plot(mae["step"], mae["traj_avg_mae_guided"], color="#d62728", alpha=0.5, lw=1.2)
ax.plot([], [], color="#1f77b4", lw=2, label="off-policy seeds")
ax.plot([], [], color="#d62728", lw=2, label="ssmc k=0 shuffled seeds")
ax.set_yscale("log")
ax.set_xlabel("training step")
ax.set_ylabel("MAE  |V_θ − V*|")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which="both")

# (d) Histogram of final MAE
ax = axes[1, 1]
ax.set_title("Final guided MAE distribution")
if off_data is not None:
    ax.hist(off_maes, bins=8, color="#1f77b4", alpha=0.6, edgecolor="white",
            label=f"off-policy n={len(off_maes)}")
ax.hist(ssmc_maes, bins=5, color="#d62728", alpha=0.6, edgecolor="white",
        label=f"ssmc k=0 shuf n={len(ssmc_maes)}")
ax.set_xlabel("final guided MAE")
ax.set_ylabel("count")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "notebooks/ssmc_shuffle_seeds.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nSaved: {out}")

with open("notebooks/ssmc_shuffle_seeds_results.json", "w") as f:
    json.dump(
        {
            "n_seeds": len(ssmc_bests),
            "best_reward": {"mean": float(ssmc_bests.mean()),
                            "sd": float(ssmc_bests.std(ddof=1)),
                            "values": ssmc_bests.tolist()},
            "final_reward": {"mean": float(ssmc_finals.mean()),
                             "sd": float(ssmc_finals.std(ddof=1)),
                             "values": ssmc_finals.tolist()},
            "final_guided_mae": {"mean": float(ssmc_maes.mean()),
                                 "sd": float(ssmc_maes.std(ddof=1)),
                                 "values": ssmc_maes.tolist()},
        },
        f, indent=2,
    )

print(f"\nTotal elapsed: {(time.time() - t_total0)/60:.1f} min")
print("Done.")
