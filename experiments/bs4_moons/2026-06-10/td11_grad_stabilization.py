#!/usr/bin/env python3
"""Gradient-stabilization study on the pinned fbrrt_td_lambda quad-only winner.

Motivation: the 2x2 probe (see REPORT) showed FBRRT instability flows through
the VALUE-GRADIENT channel -- it enters the dynamics (drift transports the
training distribution) and the targets (quadratically, via the |grad v|^2
driver) -- while pure value error is benign.  The quad-only td_lambda winner
(t11: alpha=0.96, ent=inf, grad_decay=1e-2) embraces the gradient channel and
shows good peaks (-13.1) with an unstable tail (LCB -24 +- 4).  This study
pins t11 and A/B-tests gradient-stabilizing levers, 3 seeds x 25k steps each
(long enough to see the late-training degradation that 5k-step sweeps miss).

Arms (each modifies exactly one knob unless noted):
  base      pinned t11 as-is (grad_decay=9.8e-3)
  gd0       gradient L2 penalty OFF        (ablation: does the penalty matter?)
  gd3e2     gradient L2 penalty 3e-2       (stronger)
  ema99     generation+targets read the EMA shadow (decay 0.99) -- target net
  ema999    same, decay 0.999
  clip30    drift_grad_clip 100 -> 30      (dynamics channel; bias-free)
  clip10    drift_grad_clip 100 -> 10
  drv30     driver_grad_clip = 30          (target channel; biased when active)
  drv10     driver_grad_clip = 10
  optclip   optimizer gradient_clip_val=1.0 (weight-update channel)
  warmup    alpha ramped 0 -> 0.96 over the first 2k steps
  sink      ema99 + clip10 + drv30 + optclip

Metrics per run: detrended-SEM LCB (last 20 of 50 val points), best smoothed
reward, tail plateau, diverged_at, nonfinite_skips.  Runs are idempotent
(skipped when the CSV is complete); OPT_ARM_WORKER/OPT_ARM_NWORKERS partition
the (arm, seed) grid; OPT_ARM_AGGREGATE=1 aggregates and writes
td11_stabilization_{results.json,summary.png}.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Budgets for the underlying setup (read by the exec'd sweep-script prelude).
os.environ.setdefault("OPT_MAX_STEPS", "25000")
os.environ.setdefault("OPT_N_VAL", "50")
os.environ.setdefault("OPT_LCB_TAIL", "20")
os.environ.setdefault("OPT_SKIP_NONFINITE", "1")

_setup_src = open(f"{HERE}/optuna_fbrrt_bs4_sweep.py").read()
_g = {"__name__": "td11_stab_setup",
      "__file__": f"{HERE}/optuna_fbrrt_bs4_sweep.py"}
exec(_setup_src.split("# ── Phase 1: sweep")[0], _g)

build = _g["build"]
read_curve = _g["read_curve"]
lcb_of = _g["lcb_of"]
L = _g["L"]
CSVLogger = _g["CSVLogger"]
Callback = _g["Callback"]
val_loader = _g["val_loader"]
MAX_STEPS = _g["MAX_STEPS"]
N_VAL = _g["N_VAL"]
empty_cache = _g["empty_cache"]

import gc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

# Pinned config: quad-only fbrrt_td_lambda confirmed winner t11.
_conf = json.load(open(
    f"{HERE}/optuna_fbrrt2_confirm_results_fbrrt_td_lambda_quadonly.json"))
_raw = _conf["11"]["params"]
T11 = {
    "method": "fbrrt_td_lambda",
    "n_steps": int(_raw["n_steps"]),
    "mc_samples": int(_raw["mc_samples"]),
    "branch": int(_raw["branch"]),
    "alpha": float(_raw["alpha"]),
    "off_policy_frac": float(_raw["off_policy_frac"]),
    "lr": float(_raw["lr"]),
    "loss_type": "quad",
    "entropy_lambda": float("inf"),
    "lambda_eff": float(_raw["lambda_eff"]),
    "use_grad_decay": True,
    "grad_decay": float(_raw["grad_decay"]),
}

ARMS = ["base", "gd0", "gd3e2", "ema99", "ema999", "clip30", "clip10",
        "drv30", "drv10", "optclip", "warmup", "sink"]
SEEDS = [0, 1, 2]
LOG_DIR = "lightning_logs/td11_stab"


class AlphaWarmup(Callback):
    def __init__(self, ds, target_alpha, warm_steps=2000):
        self.ds = ds; self.target = target_alpha; self.warm = warm_steps

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        f = min(1.0, trainer.global_step / self.warm)
        self.ds.fbrrt_alpha = self.target * f


def make_run(arm, seed):
    params = dict(T11)
    if arm == "gd0":
        params["use_grad_decay"] = False; params.pop("grad_decay", None)
    elif arm == "gd3e2":
        params["grad_decay"] = 3e-2
    elif arm in ("ema99", "ema999", "sink"):
        params["ema_decay"] = 0.999 if arm == "ema999" else 0.99
    model, vm, ds, loader = build(params, 3000 + seed)
    callbacks, trainer_kw = [], {}
    if arm in ("ema99", "ema999", "sink"):
        ds.value = model.ema          # generation + targets read the EMA shadow
    if arm == "clip30":
        ds.fbrrt_drift_grad_clip = 30.0
    if arm in ("clip10", "sink"):
        ds.fbrrt_drift_grad_clip = 10.0
    if arm in ("drv30", "sink"):
        ds.fbrrt_driver_grad_clip = 30.0
    if arm == "drv10":
        ds.fbrrt_driver_grad_clip = 10.0
    if arm in ("optclip", "sink"):
        trainer_kw["gradient_clip_val"] = 1.0
    if arm == "warmup":
        ds.fbrrt_alpha = 0.0
        callbacks.append(AlphaWarmup(ds, T11["alpha"], 2000))
    return model, vm, ds, loader, callbacks, trainer_kw


def run_one(arm, seed):
    name = f"{arm}_s{seed}"
    csv = f"{LOG_DIR}/{name}/version_0/metrics.csv"
    _, cv = read_curve(csv)
    if len(cv) >= int(N_VAL * 0.6):
        print(f"  {name}: cached", flush=True)
        return
    import shutil, time
    for vv in range(3):
        pth = f"{LOG_DIR}/{name}/version_{vv}"
        if os.path.exists(pth):
            shutil.rmtree(pth)
    t0 = time.time()
    model, vm, ds, loader, callbacks, trainer_kw = make_run(arm, seed)
    logger = CSVLogger(LOG_DIR, name=name, version=0)
    tr = L.Trainer(max_steps=MAX_STEPS, val_check_interval=MAX_STEPS // N_VAL,
                   logger=logger, callbacks=callbacks,
                   enable_checkpointing=False, enable_progress_bar=False,
                   **trainer_kw)
    diverged = None
    try:
        tr.fit(model, loader, val_dataloaders=val_loader)
    except (RuntimeError, ValueError) as e:
        diverged = int(tr.global_step)
        print(f"  {name}: DIVERGED at {diverged}: {str(e)[:80]}", flush=True)
    skips = int(getattr(model, "_nonfinite_count_total", 0))
    json.dump({"diverged_at": diverged, "nonfinite_skips": skips},
              open(f"{LOG_DIR}/{name}/extra.json", "w"))
    print(f"  {name}: done ({(time.time()-t0)/60:.1f} min, skips={skips}, "
          f"div={diverged})", flush=True)
    del model, vm, tr, loader, ds
    gc.collect(); empty_cache()


if os.environ.get("OPT_ARM_AGGREGATE", "0") != "1":
    wid = int(os.environ.get("OPT_ARM_WORKER", 0))
    nw = int(os.environ.get("OPT_ARM_NWORKERS", 1))
    k = 0
    for arm in ARMS:
        for seed in SEEDS:
            if k % nw == wid:
                run_one(arm, seed)
            k += 1
    print(f"[worker {wid}] done", flush=True)
    sys.exit(0)

# ── aggregate ───────────────────────────────────────────────────────────────
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rows = {}
for arm in ARMS:
    lcbs, bests, plats, divs, skipss = [], [], [], [], []
    for seed in SEEDS:
        name = f"{arm}_s{seed}"
        st, cv = read_curve(f"{LOG_DIR}/{name}/version_0/metrics.csv")
        extra = {}
        ep = f"{LOG_DIR}/{name}/extra.json"
        if os.path.exists(ep):
            extra = json.load(open(ep))
        if len(cv):
            sm = pd.Series(cv).rolling(8, min_periods=1).mean()
            lcbs.append(lcb_of(cv)); bests.append(float(sm.max()))
            plats.append(float(sm.tail(max(4, len(sm) // 5)).mean()))
        else:
            lcbs.append(-100.0); bests.append(-100.0); plats.append(-100.0)
        divs.append(extra.get("diverged_at"))
        skipss.append(extra.get("nonfinite_skips", 0))
    rows[arm] = {
        "lcb_mean": float(np.mean(lcbs)), "lcb_sd": float(np.std(lcbs)),
        "best_mean": float(np.mean(bests)), "plateau_mean": float(np.mean(plats)),
        "lcbs": lcbs, "bests": bests, "plateaus": plats,
        "diverged": divs, "skips": skipss,
        "n_diverged": sum(1 for d in divs if d is not None),
        "total_skips": int(np.sum(skipss)),
    }

print(f"\n{'arm':9s} {'LCB':>14s} {'best':>8s} {'plateau':>8s} {'#div':>5s} {'skips':>6s}")
print("-" * 56)
for arm in sorted(ARMS, key=lambda a: -rows[a]["lcb_mean"]):
    r = rows[arm]
    print(f"{arm:9s} {r['lcb_mean']:8.2f}±{r['lcb_sd']:5.2f} {r['best_mean']:8.2f} "
          f"{r['plateau_mean']:8.2f} {r['n_diverged']:>5d} {r['total_skips']:>6d}")

json.dump({"pinned_config": {k: str(v) for k, v in T11.items()},
           "max_steps": MAX_STEPS, "seeds": SEEDS, "arms": rows},
          open(f"{HERE}/td11_stabilization_results.json", "w"), indent=2)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
order = sorted(ARMS, key=lambda a: -rows[a]["lcb_mean"])
xp = np.arange(len(order))
axes[0].bar(xp, [rows[a]["lcb_mean"] for a in order],
            yerr=[rows[a]["lcb_sd"] for a in order], capsize=4, color="#16a085")
axes[0].set_xticks(xp); axes[0].set_xticklabels(order, rotation=45, fontsize=8)
axes[0].set_ylabel("LCB (3 seeds)"); axes[0].grid(True, alpha=0.3, axis="y")
axes[0].axhline(rows["base"]["lcb_mean"], color="k", ls="--", alpha=0.6,
                label="base"); axes[0].legend()
axes[0].set_title(f"t11 + stabilizers, {MAX_STEPS} steps")
for arm in order:
    col = "#c0392b" if arm == "base" else None
    for seed in SEEDS:
        st, cv = read_curve(f"{LOG_DIR}/{arm}_s{seed}/version_0/metrics.csv")
        if len(cv):
            axes[1].plot(st, pd.Series(cv).rolling(8, min_periods=1).mean(),
                         lw=1.2, alpha=0.7, color=col,
                         label=arm if seed == 0 else None)
axes[1].set_xlabel("step"); axes[1].set_ylabel("val reward (smoothed)")
axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=6, ncol=2)
axes[1].set_ylim(bottom=-40)
fig.tight_layout()
fig.savefig(f"{HERE}/td11_stabilization_summary.png", dpi=140, bbox_inches="tight")
print(f"\nSaved {HERE}/td11_stabilization_results.json + summary png")
