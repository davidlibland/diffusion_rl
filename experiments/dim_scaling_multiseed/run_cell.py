#!/usr/bin/env python3
"""One (method, dimension) cell of the multi-seed performance-by-dimension run.

Uses FIXED hyperparameters from the fitted selector (no Optuna), and evaluates
the method on `MS_N_SEEDS` independent problem instances (the GMM/reward seed),
the SAME seeds across methods so comparisons are paired.  For each seed we train
once with the selected hparams, read the converged plateau reward, and compute
regret against the analytical optimal expected reward E_{p*}[r] for that exact
(dim, seed) instance.

Results are saved incrementally to results/<method>_d<dim>.json (skips seeds
already done on restart).  Reuses sweep.build (identical training setup) and
selector.hparams_for_dim from ../dim_scaling_bs4.

Usage: python run_cell.py --method single_seed_td_lambda --dim 16
Env:   MS_N_SEEDS=30 MS_STEPS=15000 MS_VAL_EVERY=500 MS_TAIL=8 MS_TARGET_GAP=6.0
"""

import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import Callback

HERE = os.path.dirname(os.path.abspath(__file__))
BS4 = os.path.join(os.path.dirname(HERE), "dim_scaling_bs4")
sys.path.insert(0, BS4)
import sweep                       # noqa: E402  (build, val_loader, DEVICE, empty_cache)
import selector                    # noqa: E402
from problem import make_problem, optimal_terminal_and_reward  # noqa: E402

RESULTS = os.path.join(HERE, "results")
N_SEEDS = int(os.environ.get("MS_N_SEEDS", 30))
STEPS = int(os.environ.get("MS_STEPS", 15000))
VAL_EVERY = int(os.environ.get("MS_VAL_EVERY", 500))
TAIL = int(os.environ.get("MS_TAIL", 8))
TARGET_GAP = float(os.environ.get("MS_TARGET_GAP", 6.0))


class ValCollector(Callback):
    def __init__(self):
        super().__init__(); self.vals = []

    def on_validation_end(self, trainer, pl):
        m = trainer.callback_metrics.get("val_reward_mean")
        if m is not None:
            self.vals.append(float(m))


def run_seed(method, dim, s, params, hidden):
    prob = make_problem(dim, target_gap=TARGET_GAP, seed=s)
    V00, E_opt, _ = optimal_terminal_and_reward(
        prob["means"], prob["sigma2"], prob["weights"], prob["c"],
        prob["reward_scale"], dim)
    coll = ValCollector()
    try:
        model, vm, ds, loader = sweep.build(method, params, prob, dim, hidden, s)
        tr = L.Trainer(max_steps=STEPS, val_check_interval=VAL_EVERY,
                       callbacks=[coll], logger=False, enable_checkpointing=False,
                       enable_progress_bar=False, num_sanity_val_steps=0)
        tr.fit(model, loader, val_dataloaders=sweep.val_loader)
        del model, vm, tr, loader, ds
    except (RuntimeError, ValueError) as e:
        gc.collect(); sweep.empty_cache()
        return {"seed": s, "error": f"{type(e).__name__}: {str(e)[:80]}",
                "opt_reward": E_opt, "V00": V00}
    gc.collect(); sweep.empty_cache()
    vals = coll.vals
    if vals:
        sm = pd.Series(vals).rolling(min(TAIL, len(vals)), min_periods=1).mean().to_numpy()
        plateau = float(np.mean(sm[-min(TAIL, len(sm)):]))
        best, final = float(np.max(vals)), float(vals[-1])
    else:
        plateau = best = final = float("nan")
    return {"seed": s, "plateau": plateau, "best": best, "final": final,
            "opt_reward": E_opt, "V00": V00, "regret": plateau - E_opt,
            "n_val": len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dim", type=int, required=True)
    args = ap.parse_args()
    method, dim = args.method, args.dim
    hidden = min(256, max(64, 32 * dim))
    params = selector.hparams_for_dim(method, dim)
    out = f"{RESULTS}/{method}_d{dim}.json"

    data = {"method": method, "dim": dim, "hparams": params, "steps": STEPS,
            "seeds": []}
    done = set()
    if os.path.exists(out):
        data = json.load(open(out))
        data["hparams"] = params; data["steps"] = STEPS
        done = {r["seed"] for r in data["seeds"]}
    print(f"=== {method}_d{dim}  device={sweep.DEVICE}  hidden={hidden}  "
          f"done={len(done)}/{N_SEEDS} ===\n  hparams={params}", flush=True)

    for s in range(N_SEEDS):
        if s in done:
            continue
        t0 = time.time()
        r = run_seed(method, dim, s, params, hidden)
        data["seeds"].append(r)
        json.dump(data, open(out, "w"), indent=1)  # crash-safe incremental save
        msg = r.get("error") or f"plateau={r['plateau']:.3f} regret={r['regret']:+.3f}"
        print(f"  seed {s:>2}: {msg}  ({time.time()-t0:.0f}s)", flush=True)

    reg = [r["regret"] for r in data["seeds"]
           if "regret" in r and math.isfinite(r["regret"])]
    if reg:
        data["summary"] = {
            "n": len(reg), "regret_mean": float(np.mean(reg)),
            "regret_median": float(np.median(reg)),
            "regret_std": float(np.std(reg, ddof=1)) if len(reg) > 1 else 0.0,
            "regret_sem": float(np.std(reg, ddof=1) / math.sqrt(len(reg)))
            if len(reg) > 1 else 0.0,
            "plateau_mean": float(np.mean([r["plateau"] for r in data["seeds"]
                                           if "plateau" in r and math.isfinite(r["plateau"])])),
            "opt_mean": float(np.mean([r["opt_reward"] for r in data["seeds"]
                                       if math.isfinite(r["opt_reward"])])),
        }
    json.dump(data, open(out, "w"), indent=1)
    print(f"DONE {method}_d{dim}: {data.get('summary')}", flush=True)


if __name__ == "__main__":
    main()
