#!/usr/bin/env python3
"""Confirm Optuna top configs (5 seeds) then run the winners to convergence.

Phase 1 — confirm:
  6 configs (top-3 single_seed_td_lambda + top-3 single_seed_mc from
  experiments/bs4_moons/optuna_top_configs.json) × 5 seeds × 5000 steps.
  Per-seed objective = detrended-SEM LCB over the last 20 val checkpoints
  (same metric as the sweep).  Best config per method = max mean-LCB.

Phase 2 — converge:
  The 2 winners, 50000 steps, val every 1000 steps, full checkpointing
  (best.ckpt + last.ckpt — resumable for a lower-LR continuation) plus a
  plain state_dict artifact.  Convergence step is detected post-hoc from
  the rolling-mean reward curve (raw val_reward is too noisy for early-stop).
"""

import gc
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import reduce
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork


# ── Shared setup (identical to the sweep) ─────────────────────────────────
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scaler = StandardScaler(); X = scaler.fit_transform(X)
clf = GaussianMixture(n_components=100, covariance_type="spherical"); clf.fit(X)
_means = torch.from_numpy(clf.means_).double()
_sigma2 = torch.from_numpy(clf.covariances_).double()
_weights = torch.from_numpy(clf.weights_).double()
_sigmas = torch.sqrt(_sigma2)[:, None]; _weights_col = _weights[:, None]

D = 2; a = 1.0; c = torch.tensor([1.0, 0.0])
means_np = clf.means_; sigmas_np = np.sqrt(clf.covariances_); weights_np = clf.weights_


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


def make_smc_value(smc_type, k, l, model):
    if smc_type == "kt_r":
        return lambda x, t: k * t.reshape(-1) * reward_fn(x)
    if smc_type == "k_r":
        return lambda x, t: k * reward_fn(x)
    if smc_type == "k_Vema":
        return lambda x, t: k * model.ema(x, t).reshape(-1)
    if smc_type == "k_Vnograd":
        return lambda x, t: k * model.value_module(x, t).reshape(-1)
    if smc_type == "kV_plus_ltr":
        return lambda x, t: (k * model.value_module(x, t).reshape(-1)
                             + l * t.reshape(-1) * reward_fn(x))
    raise ValueError(smc_type)


# ── Constants ──────────────────────────────────────────────────────────────
LOG_DIR = "lightning_logs/optuna_confirm"
CONV_LOG_DIR = "lightning_logs/optuna_converge"
CKPT_DIR = "checkpoints/optuna_converge"
TOP_CFG = "experiments/bs4_moons/optuna_top_configs.json"

BS = 4
DS_BATCH = 64
CONFIRM_STEPS = 5000
CONFIRM_NVAL = 50
LCB_TAIL = 20
LCB_Z = 1.645
N_SEEDS = 5

CONV_STEPS = 50000
CONV_VAL_EVERY = 1000          # 50 val points over the run

all_rewards = reward_fn(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()
val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


def build(params, seed):
    L.seed_everything(seed, workers=True)
    smc_type = params["smc_type"]
    k = params["k"]
    l = params.get("l", 0.0)
    ema_decay = params.get("ema_decay", 0.99)
    grad_decay = params["grad_decay"] if params.get("use_grad_decay") else None
    vm = ValueNetwork(D, bias=bias_val)
    model = OnPolicyValueLive(
        base_score_module=base_drift, value_module=vm, reward_function=reward_fn,
        dim=D, a=a, lr=params["lr"], loss_type="quad",
        grad_decay=grad_decay, ema_decay=ema_decay,
    )
    smc_value = make_smc_value(smc_type, k, l, model)
    ds = OnPolicySMCDataset(
        dim=D, drift=base_drift, value=model.value_module,
        smc_value=smc_value, reward=reward_fn, device=DEVICE, a=a,
        batch_size=DS_BATCH, n_steps=params["n_steps"],
        mc_samples_per_step=params["mc_samples"],
        sampling_method=params["method"],
        lambda_eff=params.get("lambda_eff", 0.1),
        off_policy_frac=params["off_policy_frac"], include_t_zero=False,
        random_t=params["random_t"], generating_function=gmm_sample,
    )
    loader = DataLoader(ds, batch_size=BS)
    return model, vm, ds, loader


def read_curve(csv_path):
    if not os.path.exists(csv_path):
        return np.array([]), np.array([])
    df = pd.read_csv(csv_path)
    if "val_reward_mean" not in df.columns:
        return np.array([]), np.array([])
    sub = df.dropna(subset=["val_reward_mean"])
    return sub["step"].to_numpy(), sub["val_reward_mean"].to_numpy()


def lcb_of(curve):
    if len(curve) < 5:
        return -100.0
    tail = curve[-LCB_TAIL:]
    n = len(tail)
    xx = np.arange(n, dtype=float)
    A = np.vstack([xx, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, tail, rcond=None)
    resid = tail - A @ coef
    sigma = float(np.sqrt((resid ** 2).sum() / max(1, n - 2)))
    return float(tail.mean() - LCB_Z * sigma / np.sqrt(n))


# ── Phase 1: confirm ──────────────────────────────────────────────────────
cfgs = json.load(open(TOP_CFG))
print(f"Device: {DEVICE}")
print("PHASE 1 — confirm top-3/method × 5 seeds × 5000 steps\n", flush=True)

confirm = {}  # (method, trial) -> dict
t_start = time.time()
for method, lst in cfgs.items():
    for entry in lst:
        params = entry["params"]
        trial = entry["trial"]
        tag = f"{method}_t{trial}"
        seed_lcbs, seed_bests = [], []
        for s in range(N_SEEDS):
            name = f"{tag}_seed{s:02d}"
            csv = f"{LOG_DIR}/{name}/version_0/metrics.csv"
            if os.path.exists(csv):
                st, cv = read_curve(csv)
                if len(cv) and int(st.max()) >= CONFIRM_STEPS - 1:
                    seed_lcbs.append(lcb_of(cv)); seed_bests.append(float(cv.max()))
                    continue
            for vv in range(3):
                p = f"{LOG_DIR}/{name}/version_{vv}"
                if os.path.exists(p):
                    shutil.rmtree(p)
            t0 = time.time()
            model, vm, ds, loader = build(params, 1000 + s)
            logger = CSVLogger(LOG_DIR, name=name, version=0)
            tr = L.Trainer(
                max_steps=CONFIRM_STEPS,
                val_check_interval=max(1, CONFIRM_STEPS // CONFIRM_NVAL),
                logger=logger, enable_checkpointing=False,
                enable_progress_bar=False,
            )
            try:
                tr.fit(model, loader, val_dataloaders=val_loader)
            except RuntimeError as e:
                print(f"  {name}: error {e} -> LCB=-100", flush=True)
            del model, vm, tr, loader, ds
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            st, cv = read_curve(csv)
            lcb = lcb_of(cv) if len(cv) else -100.0
            best = float(cv.max()) if len(cv) else -100.0
            seed_lcbs.append(lcb); seed_bests.append(best)
            print(f"  {name}: LCB={lcb:.3f} best={best:.3f} "
                  f"({(time.time()-t0)/60:.1f} min)  "
                  f"[elapsed {(time.time()-t_start)/60:.1f}]", flush=True)
        seed_lcbs = np.array(seed_lcbs); seed_bests = np.array(seed_bests)
        confirm[(method, trial)] = {
            "params": params,
            "lcb_mean": float(seed_lcbs.mean()),
            "lcb_sd": float(seed_lcbs.std(ddof=1)),
            "lcb_values": seed_lcbs.tolist(),
            "best_mean": float(seed_bests.mean()),
            "best_sd": float(seed_bests.std(ddof=1)),
        }
        print(f"  >>> {tag}: LCB {seed_lcbs.mean():.3f} ± {seed_lcbs.std(ddof=1):.3f}"
              f"  best {seed_bests.mean():.3f} ± {seed_bests.std(ddof=1):.3f}",
              flush=True)

print("\n" + "=" * 88)
print(f"{'method':>22} {'trial':>5} | {'LCB mean±sd':>16} | {'best mean±sd':>16}")
print("=" * 88)
winners = {}
for method in cfgs:
    rows = [(tr, v) for (m, tr), v in confirm.items() if m == method]
    rows.sort(key=lambda r: r[1]["lcb_mean"], reverse=True)
    for tr, v in rows:
        print(f"{method:>22} {tr:>5} | "
              f"{v['lcb_mean']:>7.3f} ± {v['lcb_sd']:>5.3f} | "
              f"{v['best_mean']:>7.3f} ± {v['best_sd']:>5.3f}")
    winners[method] = rows[0]
    print(f"   -> winner {method}: trial {rows[0][0]} "
          f"(LCB {rows[0][1]['lcb_mean']:.3f})")
print("=" * 88, flush=True)

json.dump(
    {f"{m}_t{tr}": v for (m, tr), v in confirm.items()},
    open("experiments/bs4_moons/optuna_confirm_results.json", "w"), indent=2)


# ── Phase 2: converge the 2 winners ───────────────────────────────────────
print("\nPHASE 2 — run winners to convergence (50000 steps, serialized)\n",
      flush=True)


def detect_convergence(steps, curve, win=8):
    """Convergence = first step where the rolling mean reaches and stays at
    the late-run plateau (within 0.5·residual-noise)."""
    if len(curve) < win + 4:
        return None, float(curve[-1]) if len(curve) else float("nan")
    sm = pd.Series(curve).rolling(win, min_periods=1).mean().to_numpy()
    tail = sm[-max(4, len(sm) // 5):]
    plateau = float(tail.mean())
    noise = float(np.std(curve[len(curve) // 2:] - sm[len(sm) // 2:]))
    thresh = plateau - 0.5 * noise
    conv_step = None
    for i in range(len(sm)):
        if sm[i] >= thresh and np.all(sm[i:] >= plateau - noise):
            conv_step = int(steps[i]); break
    return conv_step, plateau


conv_summary = {}
for method, (trial, info) in winners.items():
    params = info["params"]
    tag = f"{method}_t{trial}_converge"
    print(f"=== {tag} ===\n  params={params}", flush=True)
    for vv in range(3):
        p = f"{CONV_LOG_DIR}/{tag}/version_{vv}"
        if os.path.exists(p):
            shutil.rmtree(p)
    ckdir = f"{CKPT_DIR}/{tag}"
    if os.path.exists(ckdir):
        shutil.rmtree(ckdir)
    os.makedirs(ckdir, exist_ok=True)

    t0 = time.time()
    model, vm, ds, loader = build(params, 20240)
    logger = CSVLogger(CONV_LOG_DIR, name=tag, version=0)
    ckpt = ModelCheckpoint(
        dirpath=ckdir, save_last=True, save_top_k=1,
        monitor="val_reward_mean", mode="max", filename="best",
    )
    tr = L.Trainer(
        max_steps=CONV_STEPS,
        val_check_interval=CONV_VAL_EVERY,
        callbacks=[ckpt], logger=logger,
        enable_checkpointing=True, enable_progress_bar=False,
    )
    tr.fit(model, loader, val_dataloaders=val_loader)
    # plain serialization for lower-LR continuation
    torch.save(
        {"state_dict": model.value_module.state_dict(),
         "params": params, "method": method, "trial": trial,
         "max_steps": CONV_STEPS},
        f"{ckdir}/value_module.pt",
    )
    st, cv = read_curve(f"{CONV_LOG_DIR}/{tag}/version_0/metrics.csv")
    conv_step, plateau = detect_convergence(st, cv)
    final_lcb = lcb_of(cv)
    print(f"  elapsed {(time.time()-t0)/60:.1f} min | plateau≈{plateau:.3f} "
          f"| converged@step={conv_step} | final-LCB={final_lcb:.3f}", flush=True)
    print(f"  ckpts: {ckdir}/best.ckpt , {ckdir}/last.ckpt , "
          f"{ckdir}/value_module.pt", flush=True)
    conv_summary[tag] = {
        "params": params, "method": method, "trial": trial,
        "plateau_reward": plateau, "convergence_step": conv_step,
        "final_lcb": final_lcb,
        "ckpt_dir": ckdir,
        "steps": st.tolist(), "val_reward": cv.tolist(),
    }
    del model, vm, tr, loader, ds
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
fig.suptitle("Optuna winners — confirmation & convergence",
             fontsize=12, fontweight="bold")

ax = axes[0]
ax.set_title("Phase 1: per-config LCB (5 seeds)")
labels, means, sds, colors = [], [], [], []
for method in cfgs:
    for (m, tr), v in confirm.items():
        if m != method:
            continue
        labels.append(f"{m[12:15]}\nt{tr}")
        means.append(v["lcb_mean"]); sds.append(v["lcb_sd"])
        colors.append("#1f77b4" if "td_lambda" in m else "#ff7f0e")
xpos = np.arange(len(labels))
ax.bar(xpos, means, yerr=sds, color=colors, capsize=4)
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("mean LCB"); ax.grid(True, alpha=0.3, axis="y")
ax.plot([], [], color="#1f77b4", lw=4, label="td_lambda")
ax.plot([], [], color="#ff7f0e", lw=4, label="mc")
ax.legend(fontsize=9)

ax = axes[1]
ax.set_title("Phase 2: convergence curves")
for tag, d in conv_summary.items():
    color = "#1f77b4" if "td_lambda" in tag else "#ff7f0e"
    st = np.array(d["steps"]); cv = np.array(d["val_reward"])
    ax.plot(st, cv, color=color, alpha=0.35, lw=1.0)
    sm = pd.Series(cv).rolling(8, min_periods=1).mean()
    ax.plot(st, sm, color=color, lw=2.0,
            label=f"{tag.split('_t')[0][12:]} (conv@{d['convergence_step']})")
    if d["convergence_step"] is not None:
        ax.axvline(d["convergence_step"], color=color, ls=":", alpha=0.7)
ax.set_xlabel("training step"); ax.set_ylabel("val reward (mean)")
ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("experiments/bs4_moons/optuna_confirm_converge.png", dpi=140, bbox_inches="tight")
print("\nSaved: experiments/bs4_moons/optuna_confirm_converge.png", flush=True)

json.dump(
    {"winners": {m: {"trial": tr, **info} for m, (tr, info) in winners.items()},
     "convergence": {k: {kk: vv for kk, vv in d.items()
                         if kk not in ("steps", "val_reward")}
                     for k, d in conv_summary.items()}},
    open("experiments/bs4_moons/optuna_confirm_converge_results.json", "w"), indent=2)
print("Saved: experiments/bs4_moons/optuna_confirm_converge_results.json")
print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min")
print("Done.")
