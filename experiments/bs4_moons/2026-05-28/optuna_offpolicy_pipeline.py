#!/usr/bin/env python3
"""Off-policy Optuna sweep → confirm → converge, batch size fixed at 4.

Mirrors the on-policy pipeline so on-vs-off is a fair comparison at BS=4.
Off-policy has no SMC knobs, so the search space is just:
    lr          loguniform 1e-4–3e-3
    loss_type   {quad, mse}
    grad_decay  {off} ∪ loguniform 1e-5–1e-1   (weight decay on ∇_x V)

Phase 1  sweep   : 40 Optuna trials, Hyperband pruning, MAX_STEPS=5000.
                   objective = detrended-SEM LCB over last 20 val checkpoints.
Phase 2  confirm : top-3 trials × 5 seeds × 5000 steps; best = max mean-LCB.
Phase 3  converge: best config, 50000 steps, full checkpointing + state_dict.

Comparison vs the on-policy winners is appended from
experiments/bs4_moons/optuna_confirm_converge_results.json if present.
"""

import gc
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from einops import reduce
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork


# ── Shared setup (identical to on-policy pipeline) ────────────────────────
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


# ── Constants (match on-policy pipeline) ──────────────────────────────────
LOG_DIR = "lightning_logs/optuna_offpolicy"
CONFIRM_DIR = "lightning_logs/optuna_off_confirm"
CONV_LOG_DIR = "lightning_logs/optuna_off_converge"
CKPT_DIR = "checkpoints/optuna_off_converge"
STUDY_DB = "sqlite:///experiments/bs4_moons/optuna_offpolicy.db"
STUDY_NAME = "offpolicy_lcb_v1"

BS = 4
MAX_STEPS = int(os.environ.get("OPT_MAX_STEPS", 5000))
N_VAL = int(os.environ.get("OPT_N_VAL", 50))
LCB_TAIL = 20
LCB_Z = 1.645
N_TRIALS = int(os.environ.get("OPT_N_TRIALS", 40))
N_SEEDS = int(os.environ.get("OPT_N_SEEDS", 5))
CONV_STEPS = int(os.environ.get("OPT_CONV_STEPS", 50000))
CONV_VAL_EVERY = int(os.environ.get("OPT_CONV_VAL_EVERY", 1000))

all_rewards = reward_fn(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()
val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


class OptunaPruning(Callback):
    def __init__(self, trial, monitor="val_reward_mean"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        m = trainer.callback_metrics.get(self.monitor)
        if m is None:
            return
        step = int(trainer.global_step)
        self.trial.report(float(m), step)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"pruned at step {step}")


def build(params, seed):
    L.seed_everything(seed, workers=True)
    grad_decay = params["grad_decay"] if params.get("use_grad_decay") else None
    vm = ValueNetwork(D, bias=bias_val)
    model = OffPolicyValue(
        base_score_module=base_drift, value_module=vm, reward_function=reward_fn,
        dim=D, a=a, lr=params["lr"], loss_type=params["loss_type"],
        grad_decay=grad_decay,
    )
    ds = InterpolatingNumpyDataset(
        generating_function=gmm_sample, a=a, batch_size=1024)
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


# ── Phase 1: sweep ────────────────────────────────────────────────────────
def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "loss_type": trial.suggest_categorical("loss_type", ["quad", "mse"]),
        "use_grad_decay": trial.suggest_categorical("use_grad_decay",
                                                    [True, False]),
    }
    if params["use_grad_decay"]:
        params["grad_decay"] = trial.suggest_float("grad_decay", 1e-5, 1e-1,
                                                   log=True)

    name = f"trial_{trial.number:04d}"
    csv = f"{LOG_DIR}/{name}/version_0/metrics.csv"
    t0 = time.time()
    model, vm, ds, loader = build(params, 1234 + trial.number)
    logger = CSVLogger(LOG_DIR, name=name, version=0)
    trainer = L.Trainer(
        max_steps=MAX_STEPS, val_check_interval=max(1, MAX_STEPS // N_VAL),
        callbacks=[OptunaPruning(trial)], logger=logger,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    try:
        trainer.fit(model, loader, val_dataloaders=val_loader)
    except optuna.TrialPruned:
        print(f"  {name}: pruned ({(time.time()-t0)/60:.1f} min)", flush=True)
        raise
    except RuntimeError as e:
        del model, vm, trainer, loader, ds
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"  {name}: error {e} -> -100", flush=True)
        return -100.0
    st, cv = read_curve(csv)
    del model, vm, trainer, loader, ds
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    lcb = lcb_of(cv)
    print(f"  {name}: LCB={lcb:.3f}  [lr={params['lr']:.1e} "
          f"loss={params['loss_type']} "
          f"gd={params.get('grad_decay','off') if params['use_grad_decay'] else 'off'}]"
          f"  {(time.time()-t0)/60:.1f} min", flush=True)
    return lcb


t_start = time.time()
sampler = TPESampler(multivariate=True, group=True, seed=42)
pruner = HyperbandPruner(min_resource=500, max_resource=MAX_STEPS,
                         reduction_factor=3)
study = optuna.create_study(
    study_name=STUDY_NAME, storage=STUDY_DB, load_if_exists=True,
    direction="maximize", sampler=sampler, pruner=pruner)
n_done = len([t for t in study.trials if t.state.is_finished()])
print(f"PHASE 1 — off-policy sweep  device={DEVICE}  "
      f"done={n_done} remaining={max(0, N_TRIALS-n_done)}", flush=True)


def _cb(study, trial):
    done = len([t for t in study.trials if t.state.is_finished()])
    try:
        best = study.best_value
    except ValueError:
        best = float("nan")
    print(f"[{done}/{N_TRIALS}] elapsed={(time.time()-t_start)/60:.1f} min "
          f"| best LCB={best:.3f}", flush=True)


study.optimize(objective, n_trials=max(0, N_TRIALS - n_done),
               callbacks=[_cb], gc_after_trial=True)

from optuna.trial import TrialState
comp = [t for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None]
comp.sort(key=lambda t: t.value, reverse=True)
top3 = comp[:3]
print("\n" + "=" * 70)
print("TOP 3 off-policy configs (by single-seed LCB):")
for t in top3:
    p = dict(t.params)
    print(f"  trial {t.number:>3}  LCB={t.value:>8.3f}  "
          f"lr={p['lr']:.2e} loss={p['loss_type']} "
          f"gd={p.get('grad_decay','off') if p.get('use_grad_decay') else 'off'}")
print("=" * 70, flush=True)
top3_cfg = [{"trial": t.number, "lcb": t.value, "params": dict(t.params)}
            for t in top3]
json.dump(top3_cfg, open("experiments/bs4_moons/optuna_offpolicy_top.json", "w"), indent=2)


# ── Phase 2: confirm (5 seeds) ────────────────────────────────────────────
print("\nPHASE 2 — confirm top-3 × 5 seeds × 5000 steps\n", flush=True)
confirm = {}
for entry in top3_cfg:
    params = entry["params"]
    trial = entry["trial"]
    seed_lcbs, seed_bests = [], []
    for s in range(N_SEEDS):
        name = f"t{trial}_seed{s:02d}"
        csv = f"{CONFIRM_DIR}/{name}/version_0/metrics.csv"
        for vv in range(3):
            p = f"{CONFIRM_DIR}/{name}/version_{vv}"
            if os.path.exists(p):
                shutil.rmtree(p)
        t0 = time.time()
        model, vm, ds, loader = build(params, 1000 + s)
        logger = CSVLogger(CONFIRM_DIR, name=name, version=0)
        tr = L.Trainer(
            max_steps=MAX_STEPS, val_check_interval=max(1, MAX_STEPS // N_VAL),
            logger=logger, enable_checkpointing=False,
            enable_progress_bar=False)
        try:
            tr.fit(model, loader, val_dataloaders=val_loader)
        except RuntimeError as e:
            print(f"  {name}: error {e}", flush=True)
        del model, vm, tr, loader, ds
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        st, cv = read_curve(csv)
        lcb = lcb_of(cv) if len(cv) else -100.0
        best = float(cv.max()) if len(cv) else -100.0
        seed_lcbs.append(lcb); seed_bests.append(best)
        print(f"  {name}: LCB={lcb:.3f} best={best:.3f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    seed_lcbs = np.array(seed_lcbs); seed_bests = np.array(seed_bests)
    confirm[trial] = {
        "params": params,
        "lcb_mean": float(seed_lcbs.mean()),
        "lcb_sd": float(seed_lcbs.std(ddof=1)),
        "lcb_values": seed_lcbs.tolist(),
        "best_mean": float(seed_bests.mean()),
        "best_sd": float(seed_bests.std(ddof=1)),
    }
    print(f"  >>> trial {trial}: LCB {seed_lcbs.mean():.3f} ± "
          f"{seed_lcbs.std(ddof=1):.3f}  best {seed_bests.mean():.3f} ± "
          f"{seed_bests.std(ddof=1):.3f}", flush=True)

best_trial = max(confirm, key=lambda tr: confirm[tr]["lcb_mean"])
best_params = confirm[best_trial]["params"]
print("\n" + "=" * 70)
print(f"OFF-POLICY winner: trial {best_trial}  "
      f"LCB {confirm[best_trial]['lcb_mean']:.3f} ± "
      f"{confirm[best_trial]['lcb_sd']:.3f}")
print(f"  params = {best_params}")
print("=" * 70, flush=True)
json.dump(confirm, open("experiments/bs4_moons/optuna_off_confirm_results.json", "w"),
          indent=2)


# ── Phase 3: converge winner ──────────────────────────────────────────────
print("\nPHASE 3 — converge off-policy winner (50000 steps, serialized)\n",
      flush=True)


def detect_convergence(steps, curve, win=8):
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


tag = f"offpolicy_t{best_trial}_converge"
for vv in range(3):
    p = f"{CONV_LOG_DIR}/{tag}/version_{vv}"
    if os.path.exists(p):
        shutil.rmtree(p)
ckdir = f"{CKPT_DIR}/{tag}"
if os.path.exists(ckdir):
    shutil.rmtree(ckdir)
os.makedirs(ckdir, exist_ok=True)

t0 = time.time()
model, vm, ds, loader = build(best_params, 20240)
logger = CSVLogger(CONV_LOG_DIR, name=tag, version=0)
ckpt = ModelCheckpoint(dirpath=ckdir, save_last=True, save_top_k=1,
                       monitor="val_reward_mean", mode="max", filename="best")
tr = L.Trainer(
    max_steps=CONV_STEPS, val_check_interval=CONV_VAL_EVERY,
    callbacks=[ckpt], logger=logger,
    enable_checkpointing=True, enable_progress_bar=False)
tr.fit(model, loader, val_dataloaders=val_loader)
torch.save({"state_dict": model.value_module.state_dict(),
            "params": best_params, "trial": best_trial,
            "max_steps": CONV_STEPS},
           f"{ckdir}/value_module.pt")
st, cv = read_curve(f"{CONV_LOG_DIR}/{tag}/version_0/metrics.csv")
conv_step, plateau = detect_convergence(st, cv)
final_lcb = lcb_of(cv)
print(f"  elapsed {(time.time()-t0)/60:.1f} min | plateau≈{plateau:.3f} | "
      f"converged@step={conv_step} | final-LCB={final_lcb:.3f}", flush=True)
print(f"  ckpts: {ckdir}/{{best.ckpt,last.ckpt,value_module.pt}}", flush=True)
del model, vm, tr, loader, ds
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()


# ── Comparison vs on-policy winners ───────────────────────────────────────
on_path = "experiments/bs4_moons/optuna_confirm_converge_results.json"
comparison = {"off_policy": {"trial": best_trial, "params": best_params,
                             "plateau_reward": plateau,
                             "convergence_step": conv_step,
                             "final_lcb": final_lcb, "ckpt_dir": ckdir}}
on_conv = {}
if os.path.exists(on_path):
    on_d = json.load(open(on_path))
    on_conv = on_d.get("convergence", {})

print("\n" + "=" * 78)
print("ON vs OFF POLICY @ BS=4  (converged)")
print("=" * 78)
print(f"{'config':>34} | {'plateau':>9} | {'conv_step':>9} | {'final_LCB':>9}")
print("-" * 78)
for k, v in on_conv.items():
    print(f"{k:>34} | {v['plateau_reward']:>9.3f} | "
          f"{str(v['convergence_step']):>9} | {v['final_lcb']:>9.3f}")
print(f"{tag:>34} | {plateau:>9.3f} | {str(conv_step):>9} | {final_lcb:>9.3f}")
print("=" * 78, flush=True)


# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
fig.suptitle("Off-policy Optuna pipeline + on-vs-off @ BS=4",
             fontsize=12, fontweight="bold")

ax = axes[0]
ax.set_title("Phase 2: off-policy top-3 LCB (5 seeds)")
labs = [f"t{tr}" for tr in confirm]
ms = [confirm[tr]["lcb_mean"] for tr in confirm]
sd = [confirm[tr]["lcb_sd"] for tr in confirm]
xp = np.arange(len(labs))
ax.bar(xp, ms, yerr=sd, color="#2ca02c", capsize=4)
ax.set_xticks(xp); ax.set_xticklabels(labs)
ax.set_ylabel("mean LCB"); ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
ax.set_title("Convergence: on (td_λ / mc) vs off-policy")
colors = {"td_lambda": "#1f77b4", "mc": "#ff7f0e", "off": "#2ca02c"}
if os.path.exists(on_path):
    for k, v in json.load(open(on_path)).get("convergence", {}).items():
        pass  # curves not stored in trimmed json; skip if absent
# off-policy curve
sm = pd.Series(cv).rolling(8, min_periods=1).mean()
ax.plot(st, cv, color=colors["off"], alpha=0.35, lw=1.0)
ax.plot(st, sm, color=colors["off"], lw=2.0,
        label=f"off-policy (conv@{conv_step})")
if conv_step is not None:
    ax.axvline(conv_step, color=colors["off"], ls=":", alpha=0.7)
# on-policy curves from their full results json (has per-step curves)
on_full = "experiments/bs4_moons/optuna_confirm_converge_results.json"
if os.path.exists(on_full):
    od = json.load(open(on_full))
    # the trimmed json lacks curves; fall back to plateau hlines
    for k, v in od.get("convergence", {}).items():
        col = colors["td_lambda"] if "td_lambda" in k else colors["mc"]
        ax.axhline(v["plateau_reward"], color=col, ls="--", alpha=0.8,
                   label=f"{k.split('_t')[0]} plateau={v['plateau_reward']:.2f}")
ax.set_xlabel("training step"); ax.set_ylabel("val reward (mean)")
ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("experiments/bs4_moons/optuna_offpolicy_pipeline.png", dpi=140,
            bbox_inches="tight")
print("\nSaved: experiments/bs4_moons/optuna_offpolicy_pipeline.png", flush=True)

json.dump(
    {"sweep_top3": [{"trial": t.number, "lcb": t.value,
                     "params": dict(t.params)} for t in top3],
     "confirm": confirm,
     "winner_trial": best_trial,
     "comparison": comparison,
     "on_policy_convergence": on_conv},
    open("experiments/bs4_moons/optuna_offpolicy_pipeline_results.json", "w"), indent=2)
print("Saved: experiments/bs4_moons/optuna_offpolicy_pipeline_results.json")
print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min")
print("Done.")
