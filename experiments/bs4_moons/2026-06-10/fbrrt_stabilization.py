#!/usr/bin/env python3
"""Apply the gradient-stabilization recipe to the STRONG FBRRT variants.

The td11 study showed the EMA generation network is the only single lever
that eliminates divergence (the gradient channel is the instability; lagging
its source breaks the net->drift->data->net feedback loop).  This study
applies it to the two competitive winners, at full 50k convergence length,
3 seeds each:

fbrrt t58 (quad-only winner, plateau -6.7 but high run-to-run variance):
  ALL of drift, driver, and bootstrap targets read one v_theta, so a slow EMA
  risks STALE BOOTSTRAP TARGETS.  Arms:
    f58_base       live net (control)
    f58_ema99      v_theta = EMA(0.99)   (smooth gradients, mild staleness)
    f58_ema999     v_theta = EMA(0.999)  (smoothest, most staleness)
    f58_hybrid     value = LIVE net, gradient = EMA(0.999) net -- the 2x2
                   probe says only the GRADIENT channel needs smoothing, so
                   this gets smooth drift/driver with zero target staleness.

fbrrt_cv t14 (best overall plateau -6.42, reproducible ~45k late death):
  Architecturally already split: the EMA is only the policy/gradient source
  (v_policy); the bootstrap target reads the LIVE net (v_target).  Slowing
  the policy EMA carries NO staleness cost.  Arms sweep the policy decay:
    cv14_base      ema_decay = 0.932 (the tuned winner)
    cv14_ema99     ema_decay = 0.99
    cv14_ema999    ema_decay = 0.999

Metrics per run: LCB / best / tail plateau + diverged_at + nonfinite skips.
Idempotent; OPT_ARM_WORKER/OPT_ARM_NWORKERS partition; OPT_ARM_AGGREGATE=1
aggregates into fbrrt_stabilization_{results.json,summary.png}.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

os.environ.setdefault("OPT_MAX_STEPS", "50000")
os.environ.setdefault("OPT_N_VAL", "50")
os.environ.setdefault("OPT_LCB_TAIL", "20")
os.environ.setdefault("OPT_SKIP_NONFINITE", "1")

_setup_src = open(f"{HERE}/optuna_fbrrt_bs4_sweep.py").read()
_g = {"__name__": "fbrrt_stab_setup",
      "__file__": f"{HERE}/optuna_fbrrt_bs4_sweep.py"}
exec(_setup_src.split("# ── Phase 1: sweep")[0], _g)

build = _g["build"]
read_curve = _g["read_curve"]
lcb_of = _g["lcb_of"]
L = _g["L"]
CSVLogger = _g["CSVLogger"]
val_loader = _g["val_loader"]
MAX_STEPS = _g["MAX_STEPS"]
N_VAL = _g["N_VAL"]
empty_cache = _g["empty_cache"]

import gc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402


def _winner(fname, key, method):
    raw = json.load(open(f"{HERE}/{fname}"))[key]["params"]
    p = {
        "method": method,
        "n_steps": int(raw["n_steps"]),
        "mc_samples": int(raw["mc_samples"]),
        "branch": int(raw["branch"]),
        "alpha": float(raw["alpha"]),
        "off_policy_frac": float(raw["off_policy_frac"]),
        "lr": float(raw["lr"]),
        "loss_type": raw["loss_type"],
        "entropy_lambda": (float("inf") if raw["entropy_lambda"] == "inf"
                           else float(raw["entropy_lambda"])),
        "use_grad_decay": bool(raw.get("use_grad_decay")),
    }
    if p["use_grad_decay"]:
        p["grad_decay"] = float(raw["grad_decay"])
    if raw.get("ema_decay") is not None:
        p["ema_decay"] = float(raw["ema_decay"])
    if "lambda_eff" in raw:
        p["lambda_eff"] = float(raw["lambda_eff"])
    return p


T58 = _winner("optuna_fbrrt2_confirm_results_fbrrt_quadonly.json", "58", "fbrrt")
T14 = _winner("optuna_fbrrt2_confirm_results_fbrrt_cv.json", "14", "fbrrt_cv")


class HybridValue:
    """value = live net, gradient (w.r.t. x) = EMA net.

    v(x) = live(x).detach() + ema(x) - ema(x).detach():  the VALUE equals the
    live network's output, while autograd through x flows only through the
    EMA term -- smooth drift/driver gradients with zero bootstrap staleness.
    """

    def __init__(self, live, ema):
        self.live = live; self.ema = ema

    def __call__(self, x, t):
        ev = self.ema(x, t)
        return self.live(x, t).detach() + ev - ev.detach()


ARMS = ["f58_base", "f58_ema99", "f58_ema999", "f58_hybrid",
        "cv14_base", "cv14_ema99", "cv14_ema999"]
SEEDS = [0, 1, 2]
LOG_DIR = "lightning_logs/fbrrt_stab"


def make_run(arm, seed):
    if arm.startswith("f58"):
        params = dict(T58)
        if arm in ("f58_ema99",):
            params["ema_decay"] = 0.99
        elif arm in ("f58_ema999", "f58_hybrid"):
            params["ema_decay"] = 0.999
    else:
        params = dict(T14)
        if arm == "cv14_ema99":
            params["ema_decay"] = 0.99
        elif arm == "cv14_ema999":
            params["ema_decay"] = 0.999
    model, vm, ds, loader = build(params, 4000 + seed)
    if arm in ("f58_ema99", "f58_ema999"):
        ds.value = model.ema                       # full target network
    elif arm == "f58_hybrid":
        ds.value = HybridValue(model.value_module, model.ema)
    # cv arms: build() already wires smc_value (= v_policy) to model.ema with
    # the requested decay; v_target stays the live net.
    return model, vm, ds, loader


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
    model, vm, ds, loader = make_run(arm, seed)
    logger = CSVLogger(LOG_DIR, name=name, version=0)
    tr = L.Trainer(max_steps=MAX_STEPS, val_check_interval=MAX_STEPS // N_VAL,
                   logger=logger, enable_checkpointing=False,
                   enable_progress_bar=False)
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
        "best_mean": float(np.mean(bests)),
        "plateau_mean": float(np.mean(plats)),
        "plateau_sd": float(np.std(plats)),
        "lcbs": lcbs, "bests": bests, "plateaus": plats,
        "diverged": divs, "skips": skipss,
        "n_diverged": sum(1 for d in divs if d is not None),
    }

print(f"\n{'arm':12s} {'plateau':>14s} {'best':>8s} {'LCB':>8s} {'#div':>5s} {'skips':>6s}")
print("-" * 60)
for arm in ARMS:
    r = rows[arm]
    print(f"{arm:12s} {r['plateau_mean']:8.2f}±{r['plateau_sd']:5.2f} "
          f"{r['best_mean']:8.2f} {r['lcb_mean']:8.2f} {r['n_diverged']:>5d} "
          f"{int(np.sum(r['skips'])):>6d}")

json.dump({"t58": {k: str(v) for k, v in T58.items()},
           "t14": {k: str(v) for k, v in T14.items()},
           "max_steps": MAX_STEPS, "seeds": SEEDS, "arms": rows},
          open(f"{HERE}/fbrrt_stabilization_results.json", "w"), indent=2)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for ax, prefix, title in [(axes[0], "f58", "fbrrt t58"),
                          (axes[1], "cv14", "fbrrt_cv t14")]:
    for arm in [a for a in ARMS if a.startswith(prefix)]:
        for seed in SEEDS:
            st, cv = read_curve(f"{LOG_DIR}/{arm}_s{seed}/version_0/metrics.csv")
            if len(cv):
                ax.plot(st, pd.Series(cv).rolling(8, min_periods=1).mean(),
                        lw=1.2, alpha=0.75, label=arm if seed == 0 else None)
    ax.set_title(f"{title} -- EMA stabilization, 3 seeds x {MAX_STEPS}")
    ax.set_xlabel("step"); ax.set_ylabel("val reward (smoothed)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8); ax.set_ylim(bottom=-30)
fig.tight_layout()
fig.savefig(f"{HERE}/fbrrt_stabilization_summary.png", dpi=140,
            bbox_inches="tight")
print(f"\nSaved {HERE}/fbrrt_stabilization_results.json + summary png")
