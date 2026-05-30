"""
Mixed off-policy + on-policy training.

Each training step draws from either:
  - Off-policy: (r(x1), x_t, t) with x1 ~ p1 (correct targets, broad coverage)
  - On-policy:  (V_ema(x_next), x, t) via ATD(λ=0) + smc=reward (guided coverage)

The off-policy data provides a stabilizing anchor (always-correct targets),
while on-policy data provides samples from the guided distribution.

Configurations:
  - offpolicy (baseline)
  - osb_blend ws=3000 (previous best)
  - mixed 50/50 with EMA decay=0.999, ws=3000
  - mixed 50/50 with EMA decay=0.99, ws=3000
  - mixed 25/75 (25% off, 75% on) with EMA decay=0.999, ws=3000
  - pure on-policy with EMA decay=0.999, ws=3000 (for comparison)
"""

import json
import os
import shutil
import time
from functools import partial
from math import sqrt

import matplotlib

matplotlib.use("Agg")
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import reduce
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

# ---------------------------------------------------------------------------
# GMM Setup
# ---------------------------------------------------------------------------
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler()
X = scalar.fit_transform(X)
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


class AnalyticalValue(nn.Module):
    def __init__(self, means, sigma2, weights, a=1.0, c=None, D=2):
        super().__init__()
        if c is None:
            c = torch.tensor([1.0, 0.0])
        self.register_buffer("means", means.float())
        self.register_buffer("sigma2", sigma2.float())
        self.register_buffer("weights", weights.float())
        self.a = a
        self.D = D
        self.register_buffer("c", c.float())

    def _log_Z(self, m, v):
        cc = self.c.double()
        denom = 1.0 + 20.0 * v
        return (
            -self.D / 2.0 * torch.log(denom)
            + (-10 * (m**2).sum(-1) + 20 * (m * cc).sum(-1) + 200 * v * (cc**2).sum())
            / denom
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
        lg = -self.D / 2.0 * torch.log(2 * torch.pi * ts * dk) - d2 / (2 * ts * dk)
        lw = torch.log(w)[None, :]
        lpw = lw + lg - torch.logsumexp(lw + lg, dim=1, keepdim=True)
        tV = 2 * self.a * (1 - t_) * s2[None, :] / dk
        tmu = (
            s2[None, :, None] * x[:, None, :]
            + 2 * self.a * (1 - t_)[:, :, None] * m[None, :, :]
        ) / dk[:, :, None]
        return torch.logsumexp(lpw + self._log_Z(tmu, tV), dim=1).float()


_anal_vm_cpu = AnalyticalValue(_means, _sigma2, _weights, a=a, c=c, D=D)


def anal_fn(x, t):
    return _anal_vm_cpu(x.cpu(), t.cpu()).to(x.device)


def get_conditional_mixture(xt, ts, means, sigmas, weights, a):
    n, d = xt.shape
    xt_ = xt[..., None]
    means_ = means.T[None, ...]
    ts_ = ts[..., None]
    sigmas_ = sigmas.T
    weights_ = weights.T
    olw = torch.log(weights_)
    denom = 2 * a * (1 - ts) + ts * sigmas_**2
    le = -reduce((xt_ - means_ * ts_) ** 2, "n d m -> n m", "sum") / (2 * ts * denom)
    lsf = torch.log(2 * a * (1 - ts) / denom) * d / 2
    lrw = olw + le + lsf
    lw = lrw - torch.logsumexp(lrw, dim=1, keepdim=True)
    lw = torch.where((ts == 0), olw, lw)
    nm = (2 * a * (1 - ts_) * means_ + xt_ * sigmas_[None, ...] ** 2) / denom[
        :, None, :
    ]
    return {"log_weights": lw, "means": nm}


def gmm_drift(xt, ts, a):
    ts = ts.reshape(-1, 1)
    cond = get_conditional_mixture(
        xt,
        ts,
        _means.float().to(xt),
        _sigmas.float().to(xt),
        _weights_col.float().to(xt),
        a,
    )
    us = (cond["means"] - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(
        torch.exp(cond["log_weights"])[:, None, :] * us, "n d m -> n d", "sum"
    )


DEVICE = "mps"
base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(
    dtype=torch.float
)
reward_fn = lambda x: -10 * (x - c.to(x)).square().sum(dim=1)
smc_reward = lambda x, t: reward_fn(x)


def gmm_sample(n):
    k = np.random.choice(len(weights_np), size=n, p=weights_np)
    return means_np[k] + sigmas_np[k, np.newaxis] * np.random.randn(n, D)


with open("experiments/common/analytical_target.json") as f:
    _at = json.load(f)
E_OPT = _at["E_opt"]
all_rewards = reward_fn(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()
print(f"E_OPT = {E_OPT:.4f}")


# ---------------------------------------------------------------------------
# Mixed dataset: interleaves off-policy and on-policy samples
# ---------------------------------------------------------------------------
class MixedDataset(IterableDataset):
    """Yields (target, x, t) from a mix of off-policy and on-policy sources.

    off_policy_frac: fraction of samples from off-policy [0, 1].
    """

    def __init__(self, offpolicy_ds, onpolicy_ds, reward, off_frac=0.5):
        self.offpolicy_ds = offpolicy_ds
        self.onpolicy_ds = onpolicy_ds
        self.reward = reward
        self.off_frac = off_frac

    def __iter__(self):
        off_iter = iter(self.offpolicy_ds)
        on_iter = iter(self.onpolicy_ds)
        while True:
            if torch.rand(1).item() < self.off_frac:
                # Off-policy: yields (x1, x, t) → convert to (r(x1), x, t)
                x1, x, t = next(off_iter)
                target = self.reward(x1.unsqueeze(0)).squeeze(0)
                yield target.unsqueeze(-1), x, t
            else:
                # On-policy: already yields (target, x, t)
                yield next(on_iter)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LR = 3e-3
LOADER_BATCH_SIZE = 256
WS_STEPS = 3000
ON_STEPS = 3000
TOTAL_STEPS = WS_STEPS + ON_STEPS
LOG_DIR = "lightning_logs/mixed_exp"
CKPT_DIR = "checkpoints/mixed_exp"

BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_NAMES = ["t00_20", "t20_40", "t40_60", "t60_80", "t80_100"]

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


# ---------------------------------------------------------------------------
# Trajectory eval callback
# ---------------------------------------------------------------------------
class TrajectoryEvalCallback(Callback):
    def __init__(self, anal_fn, n_traj=256, n_steps=100):
        super().__init__()
        self.anal_fn = anal_fn
        self.n_traj = n_traj
        self.n_steps = n_steps

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx > 0:
            return
        dim = pl_module.hparams.dim
        device = pl_module.device
        n = self.n_traj
        dt = 1.0 / self.n_steps
        for beta, label in [(0, "base"), (1, "guided")]:
            x = torch.zeros(n, dim, device=device)
            all_x, all_t = [x], [torch.zeros(n, device=device)]
            drift_fn = partial(pl_module.drift, beta=beta)
            for step_t in torch.linspace(0, 1, self.n_steps + 1, device=device)[:-1]:
                t_vec = step_t.expand(n)
                dx = drift_fn(x, t_vec) * dt
                db = sqrt(2 * pl_module.a * dt) * torch.randn_like(x)
                x = x + dx + db
                all_x.append(x)
                all_t.append(torch.full((n,), float(step_t) + dt, device=device))
            all_x = torch.cat(all_x)
            all_t = torch.cat(all_t)
            with torch.no_grad():
                v_pred = pl_module.value_module(all_x, all_t)
                v_anal = self.anal_fn(all_x, all_t)
            err = v_pred - v_anal
            pl_module.log(f"traj_avg_mae_{label}", err.abs().mean(), prog_bar=False)
            pl_module.log(f"traj_avg_bias_{label}", err.mean(), prog_bar=False)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_offpolicy(vm, max_steps, run_name, version=0):
    ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=1024)
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
    model = OffPolicyValue(
        base_score_module=base_drift,
        reward_function=reward_fn,
        value_module=vm,
        dim=D,
        a=a,
        lr=LR,
        loss_type="quad",
        analytical_value_fn=anal_fn,
    )
    traj_cb = TrajectoryEvalCallback(anal_fn)
    logger = CSVLogger(LOG_DIR, name=run_name, version=version)
    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=max(1, max_steps // 60),
        callbacks=[traj_cb],
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    return vm


def train_mixed(vm, max_steps, run_name, off_frac, ema_decay, version=1):
    """Mixed off-policy + on-policy with EMA value."""
    model = OnPolicyValue(
        base_score_module=base_drift,
        value_module=vm,
        reward_function=reward_fn,
        dim=D,
        a=a,
        lr=LR,
        loss_type="quad",
        analytical_value_fn=anal_fn,
        ema_decay=ema_decay,
    )

    offpolicy_ds = InterpolatingNumpyDataset(
        generating_function=gmm_sample,
        a=a,
        batch_size=1024,
    )
    onpolicy_ds = OnPolicySMCDataset(
        dim=D,
        drift=base_drift,
        value=model.ema,
        smc_value=smc_reward,
        reward=reward_fn,
        device=DEVICE,
        a=a,
        batch_size=1,
        n_steps=100,
        mc_samples_per_step=10,
        sampling_method="ancestral_td_lambda",
        lambda_eff=0.0,
    )
    mixed_ds = MixedDataset(offpolicy_ds, onpolicy_ds, reward_fn, off_frac=off_frac)
    loader = DataLoader(mixed_ds, batch_size=LOADER_BATCH_SIZE)

    traj_cb = TrajectoryEvalCallback(anal_fn)
    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{run_name}",
        save_last=True,
        save_top_k=1,
        monitor="val_reward_mean",
        mode="max",
        filename="best",
    )
    logger = CSVLogger(LOG_DIR, name=run_name, version=version)
    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=max(1, max_steps // 60),
        callbacks=[ckpt_cb, traj_cb],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    return vm


def train_pure_onpolicy_ema(vm, max_steps, run_name, ema_decay, version=1):
    """Pure on-policy ATD(λ=0) + smc=reward + EMA value."""
    model = OnPolicyValue(
        base_score_module=base_drift,
        value_module=vm,
        reward_function=reward_fn,
        dim=D,
        a=a,
        lr=LR,
        loss_type="quad",
        analytical_value_fn=anal_fn,
        ema_decay=ema_decay,
    )
    ds = OnPolicySMCDataset(
        dim=D,
        drift=base_drift,
        value=model.ema,
        smc_value=smc_reward,
        reward=reward_fn,
        device=DEVICE,
        a=a,
        batch_size=32,
        n_steps=100,
        mc_samples_per_step=10,
        sampling_method="ancestral_td_lambda",
        lambda_eff=0.0,
    )
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
    traj_cb = TrajectoryEvalCallback(anal_fn)
    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{run_name}",
        save_last=True,
        save_top_k=1,
        monitor="val_reward_mean",
        mode="max",
        filename="best",
    )
    logger = CSVLogger(LOG_DIR, name=run_name, version=version)
    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=max(1, max_steps // 60),
        callbacks=[ckpt_cb, traj_cb],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    return vm


def train_osb_blend(vm, max_steps, run_name, version=1):
    """OSB + blended V + smc=reward (previous best)."""
    model = OnPolicyValue(
        base_score_module=base_drift,
        value_module=vm,
        reward_function=reward_fn,
        dim=D,
        a=a,
        lr=LR,
        loss_type="quad",
        analytical_value_fn=anal_fn,
    )
    ds = OnPolicySMCDataset(
        dim=D,
        drift=base_drift,
        value=vm,
        smc_value=smc_reward,
        reward=reward_fn,
        device=DEVICE,
        a=a,
        batch_size=32,
        n_steps=100,
        mc_samples_per_step=10,
        sampling_method="one_step_bootstrap",
        lambda_eff=0.0,
    )
    ds.raw_value_fn = ds.value_fn  # monkey-patch: use blend
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
    traj_cb = TrajectoryEvalCallback(anal_fn)
    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{run_name}",
        save_last=True,
        save_top_k=1,
        monitor="val_reward_mean",
        mode="max",
        filename="best",
    )
    logger = CSVLogger(LOG_DIR, name=run_name, version=version)
    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=max(1, max_steps // 60),
        callbacks=[ckpt_cb, traj_cb],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    return vm


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
RUNS = [
    dict(name="offpolicy", method="offpolicy"),
    dict(name="osb_blend", method="osb_blend"),
    dict(name="mixed50_ema999", method="mixed", off_frac=0.5, ema_decay=0.999),
    dict(name="mixed50_ema99", method="mixed", off_frac=0.5, ema_decay=0.99),
    dict(name="mixed25_ema999", method="mixed", off_frac=0.25, ema_decay=0.999),
    dict(name="pure_ema999", method="pure_ema", ema_decay=0.999),
]


for cfg in RUNS:
    run_name = cfg["name"]
    method = cfg["method"]

    print(f"\n{'=' * 70}")
    if method == "offpolicy":
        print(f"  {run_name}: {TOTAL_STEPS} off-policy steps")
    elif method == "mixed":
        print(f"  {run_name}: {WS_STEPS} off → {ON_STEPS} mixed")
        print(f"    off_frac={cfg['off_frac']} ema_decay={cfg['ema_decay']}")
    elif method == "pure_ema":
        print(
            f"  {run_name}: {WS_STEPS} off → {ON_STEPS} on-policy (EMA decay={cfg['ema_decay']})"
        )
    else:
        print(f"  {run_name}: {WS_STEPS} off → {ON_STEPS} on ({method})")
    print(f"{'=' * 70}")

    # Check completion
    if method == "offpolicy":
        csv_check = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
        expected = TOTAL_STEPS
    else:
        csv_check = f"{LOG_DIR}/{run_name}/version_1/metrics.csv"
        expected = ON_STEPS

    if os.path.exists(csv_check):
        df = pd.read_csv(csv_check)
        val = df.dropna(subset=["val_reward_mean"])
        if len(val) > 0 and val["step"].max() >= expected - 1:
            print("  Already complete, skipping.")
            continue

    for v in range(3):
        p = f"{LOG_DIR}/{run_name}/version_{v}"
        if os.path.exists(p):
            shutil.rmtree(p)

    vm = ValueNetwork(D, bias=bias_val)

    if method == "offpolicy":
        t0 = time.perf_counter()
        train_offpolicy(vm, TOTAL_STEPS, run_name)
        print(f"  Elapsed: {(time.perf_counter() - t0) / 60:.1f} min")
        continue

    # Phase 1: off-policy warm-start
    print(f"  Phase 1: Off-policy for {WS_STEPS} steps...")
    t0 = time.perf_counter()
    train_offpolicy(vm, WS_STEPS, run_name, version=0)
    print(f"  Phase 1 done: {(time.perf_counter() - t0) / 60:.1f} min")

    # Phase 2
    print(f"  Phase 2: {method} for {ON_STEPS} steps...")
    t0 = time.perf_counter()
    if method == "mixed":
        train_mixed(vm, ON_STEPS, run_name, cfg["off_frac"], cfg["ema_decay"])
    elif method == "pure_ema":
        train_pure_onpolicy_ema(vm, ON_STEPS, run_name, cfg["ema_decay"])
    elif method == "osb_blend":
        train_osb_blend(vm, ON_STEPS, run_name)
    print(f"  Phase 2 done: {(time.perf_counter() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\n\n{'=' * 70}")
print(f"  RESULTS (E_OPT = {E_OPT:.4f})")
print(f"{'=' * 70}")


def load_combined(run_name, method):
    dfs = []
    csv0 = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
    if os.path.exists(csv0):
        dfs.append(pd.read_csv(csv0))
    if method != "offpolicy":
        csv1 = f"{LOG_DIR}/{run_name}/version_1/metrics.csv"
        if os.path.exists(csv1):
            d = pd.read_csv(csv1).copy()
            d["step"] = d["step"] + WS_STEPS
            dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else None


print(
    f"\n  {'Run':<25} {'Best Rwd':>9} {'Final Rwd':>10} {'Gap':>7} {'Base MAE':>9} {'Guided MAE':>11}"
)
print(f"  {'-' * 75}")

run_data = {}
for cfg in RUNS:
    rn = cfg["name"]
    df = load_combined(rn, cfg["method"])
    if df is None:
        continue
    val = df.dropna(subset=["val_reward_mean"])
    if len(val) == 0:
        continue
    best = val["val_reward_mean"].max()
    final = val["val_reward_mean"].iloc[-1]
    gap = E_OPT - best
    bm = df.dropna(subset=["traj_avg_mae_base"])
    gm = df.dropna(subset=["traj_avg_mae_guided"])
    base_f = bm["traj_avg_mae_base"].iloc[-1] if len(bm) > 0 else float("nan")
    guid_f = gm["traj_avg_mae_guided"].iloc[-1] if len(gm) > 0 else float("nan")
    print(
        f"  {rn:<25} {best:>9.4f} {final:>10.4f} {gap:>7.4f} {base_f:>9.4f} {guid_f:>11.4f}"
    )
    run_data[rn] = df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Mixed Training Experiment", fontsize=14, fontweight="bold")

cmap = plt.cm.tab10
colors = {}
for i, cfg in enumerate(RUNS):
    colors[cfg["name"]] = cmap(i / len(RUNS))
colors["offpolicy"] = "black"
colors["osb_blend"] = "red"

# Plot 1: Terminal reward
ax = axes[0]
ax.set_title("Terminal Reward vs Steps")
for cfg in RUNS:
    rn = cfg["name"]
    if rn not in run_data:
        continue
    df = run_data[rn]
    val = df.dropna(subset=["val_reward_mean"])
    ls = "--" if rn == "offpolicy" else (":" if "osb" in rn else "-")
    lw = 2.5 if rn in ("offpolicy", "osb_blend") else 1.5
    ax.plot(
        val["step"].values,
        val["val_reward_mean"].values,
        color=colors[rn],
        linestyle=ls,
        linewidth=lw,
        label=rn,
    )
ax.axhline(
    E_OPT,
    color="red",
    linestyle=":",
    linewidth=1,
    alpha=0.5,
    label=f"E_opt={E_OPT:.3f}",
)
ax.axvline(WS_STEPS, color="gray", linestyle=":", alpha=0.3)
ax.set_xlabel("Steps")
ax.set_ylabel("Avg Terminal Reward")
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)

# Plot 2: Base drift MAE
ax = axes[1]
ax.set_title("V Error: Base Drift (β=0)")
for cfg in RUNS:
    rn = cfg["name"]
    if rn not in run_data:
        continue
    df = run_data[rn]
    sub = df.dropna(subset=["traj_avg_mae_base"])
    if len(sub) == 0:
        continue
    ls = "--" if rn == "offpolicy" else (":" if "osb" in rn else "-")
    lw = 2.5 if rn in ("offpolicy", "osb_blend") else 1.5
    ax.plot(
        sub["step"].values,
        sub["traj_avg_mae_base"].values,
        color=colors[rn],
        linestyle=ls,
        linewidth=lw,
        label=rn,
    )
ax.set_xlabel("Steps")
ax.set_ylabel("Avg MAE")
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

# Plot 3: Guided MAE
ax = axes[2]
ax.set_title("V Error: Guided (β=1)")
for cfg in RUNS:
    rn = cfg["name"]
    if rn not in run_data:
        continue
    df = run_data[rn]
    sub = df.dropna(subset=["traj_avg_mae_guided"])
    if len(sub) == 0:
        continue
    ls = "--" if rn == "offpolicy" else (":" if "osb" in rn else "-")
    lw = 2.5 if rn in ("offpolicy", "osb_blend") else 1.5
    ax.plot(
        sub["step"].values,
        sub["traj_avg_mae_guided"].values,
        color=colors[rn],
        linestyle=ls,
        linewidth=lw,
        label=rn,
    )
ax.set_xlabel("Steps")
ax.set_ylabel("Avg MAE")
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

plt.tight_layout()
plt.savefig("experiments/misc/2026-05-06_mixed_training/mixed_training_exp.png", dpi=150, bbox_inches="tight")
print("\nSaved: experiments/misc/2026-05-06_mixed_training/mixed_training_exp.png")
plt.close()

print(f"\nDone. E_OPT = {E_OPT:.4f}")
