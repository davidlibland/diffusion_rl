#!/usr/bin/env python3
"""One (method, dimension) cell of the law-v2 multi-seed grid.

Hyperparameters (including the recipe knobs n_steps / expand_frac /
epoch_rows) served per dimension by fit_lawv2.hparams_for_dim; otherwise
identical protocol to ../dim_scaling_consth/run_cell.py (constant 6-nat
headroom, coordinate-nested paired instances, 15k steps, 30 seeds,
frac_closed = (plateau - E_base)/6).

Usage: python run_lawv2_cell.py --method single_seed_mc --dim 64
Env:   DSR_N_SEEDS=30 DSC_STEPS=15000 DSC_VAL_EVERY=500 DSC_TAIL=8
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "dim_scaling_bs4"))
sys.path.insert(0, HERE)

import sweep_lawv2 as sw2  # noqa: E402
import sweep_consth as base  # noqa: E402  (frozen local copy)
from fit_lawv2 import hparams_for_dim  # noqa: E402
from problem_consth import make_problem  # noqa: E402
from problem import optimal_terminal_and_reward  # noqa: E402

RESULTS = os.environ.get("DSR_RESULTS_DIR", os.path.join(HERE, "results"))
N_SEEDS = int(os.environ.get("DSR_N_SEEDS", 30))
STEPS = int(os.environ.get("DSC_STEPS", 15000))
VAL_EVERY = int(os.environ.get("DSC_VAL_EVERY", 500))
TAIL = int(os.environ.get("DSC_TAIL", 8))


class ValCollector(Callback):
    def __init__(self):
        super().__init__(); self.vals = []

    def on_validation_end(self, trainer, pl):
        m = trainer.callback_metrics.get("val_reward_mean")
        if m is not None:
            self.vals.append(float(m))


def run_seed(method, dim, s, params, hidden):
    prob = make_problem(dim, seed=s)
    _, e_opt, _ = optimal_terminal_and_reward(
        prob["means"], prob["sigma2"], prob["weights"], prob["c"],
        prob["reward_scale"], dim)
    model, vm, ds, loader = sw2.build(method, params, prob, dim, hidden, s)
    vc = ValCollector()
    tr = L.Trainer(max_steps=STEPS, val_check_interval=VAL_EVERY,
                   callbacks=[vc], logger=False, enable_checkpointing=False,
                   enable_progress_bar=False, num_sanity_val_steps=0)
    err = None
    try:
        tr.fit(model, loader, val_dataloaders=base.val_loader)
    except (RuntimeError, ValueError) as e:
        err = f"{type(e).__name__}: {str(e)[:80]}"
    v = np.array(vc.vals, dtype=float)
    if len(v):
        sm = pd.Series(v).rolling(TAIL, min_periods=1).mean().to_numpy()
        plateau = float(np.nanmean(sm[-TAIL:])); best = float(np.nanmax(sm))
    else:
        plateau = best = float("nan")
    out = {"seed": s, "plateau": plateau, "best": best,
           "opt_reward": float(e_opt), "E_base": prob["diag"]["E_base_r"],
           "regret": plateau - float(e_opt),
           "frac_closed": ((plateau - prob["diag"]["E_base_r"]) / 6.0
                           if math.isfinite(plateau) else float("nan")),
           "nonfinite_skips": int(getattr(model, "_nonfinite_count_total", 0)),
           "error": err}
    del model, vm, ds, loader, tr
    gc.collect(); base.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dim", type=int, required=True)
    args = ap.parse_args()
    method, dim = args.method, args.dim
    os.makedirs(RESULTS, exist_ok=True)
    out_path = f"{RESULTS}/gridv2_{method}_d{dim}.json"
    params = hparams_for_dim(method, dim)
    hidden = min(256, max(64, 32 * dim))
    rec = (json.load(open(out_path)) if os.path.exists(out_path) else
           {"method": method, "dim": dim, "steps": STEPS, "hparams": params,
            "seeds": []})
    done = {e["seed"] for e in rec["seeds"]}
    print(f"=== GRIDV2 {method}_d{dim} done={len(done)}/{N_SEEDS} "
          f"[{sw2.fmt(method, params)}]", flush=True)
    for s in range(N_SEEDS):
        if s in done:
            continue
        t0 = time.time()
        out = run_seed(method, dim, s, params, hidden)
        rec["seeds"].append(out)
        json.dump(rec, open(out_path, "w"), indent=1)
        print(f"  seed {s:2d}: plateau={out['plateau']:8.3f} "
              f"closed={100*out['frac_closed']:5.1f}% err={out['error']} "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)
    fr = np.array([e["frac_closed"] for e in rec["seeds"]
                   if math.isfinite(e.get("frac_closed", float("nan")))])
    if len(fr):
        print(f"  CELL DONE frac_closed={100*fr.mean():.1f}%±"
              f"{100*fr.std(ddof=1)/math.sqrt(len(fr)):.1f} (n={len(fr)})",
              flush=True)


if __name__ == "__main__":
    main()
