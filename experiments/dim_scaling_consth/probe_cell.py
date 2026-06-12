#!/usr/bin/env python3
"""Failure-mode probe for the SMC methods at high dimension.

The grid showed every SMC method decaying with d while off-policy stays flat,
and the anchor sweeps were too flat at 5k steps to attribute a cause.  This
probe re-runs the SMC methods at FULL grid length (15k steps) on a paired
subset of the grid seeds, changing exactly ONE factor per arm relative to the
law-fitted control (= the existing grid cells):

  reward_twist : the OLD study's fixed-reward twist (signal from step 0):
                 ssmc -> kt_r k=0.0071, ssmc-td -> k_r k=0.1745,
                 anc-mc-td -> kt_r k=0.0170.  Tests "value-based twists are
                 uninformative early / never selected for high-d".
  mc16         : law config with mc_samples=16 (vs 9/9/1).  Tests particle
                 starvation against the e^{-gap}-rare tilted region
                 (especially anc-mc-td's degenerate mc=1).
  guid_toggle  : law config with use_guidance flipped (ssmc on->off,
                 ssmc-td off->on @0.61/ema, anc-mc-td on->off).  The 15k-step
                 version of the driver question.

Results: results/probe_<arm>_<method>_d<dim>.json (same schema as grid cells,
seeds 0..DSC_PROBE_SEEDS-1 paired with the grid).

Usage: python probe_cell.py --method single_seed_mc --dim 512 --arm mc16
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "dim_scaling_bs4"))
sys.path.insert(0, HERE)

import run_cell as rc  # noqa: E402  (run_seed; main() is __name__-guarded)
import sweep_consth as sw  # noqa: E402
from fit_consth import hparams_for_dim  # noqa: E402

RESULTS = os.environ.get("DSC_RESULTS_DIR", os.path.join(HERE, "results"))
N_SEEDS = int(os.environ.get("DSC_PROBE_SEEDS", 10))

OLD_REWARD_TWIST = {
    "single_seed_mc": ("kt_r", 0.0071),
    "single_seed_td_lambda": ("k_r", 0.1745),
    "ancestral_mc_td_lambda": ("kt_r", 0.0170),
}


def arm_params(method, dim, arm):
    p = hparams_for_dim(method, dim)
    if arm == "reward_twist":
        smc_type, k = OLD_REWARD_TWIST[method]
        p["smc_type"] = smc_type
        p["k"] = k
        p.pop("l", None)
    elif arm == "mc16":
        p["mc_samples"] = 16
    elif arm == "guid_toggle":
        if p.get("use_guidance"):
            p["use_guidance"] = False
            p.pop("guidance_scale", None)
            p.pop("guidance_source", None)
        else:
            p["use_guidance"] = True
            p["guidance_scale"] = 0.61
            p["guidance_source"] = "ema"
            p.setdefault("ema_decay", 0.99)
    else:
        raise ValueError(arm)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--arm", required=True,
                    choices=["reward_twist", "mc16", "guid_toggle"])
    args = ap.parse_args()
    method, dim, arm = args.method, args.dim, args.arm
    os.makedirs(RESULTS, exist_ok=True)
    out_path = f"{RESULTS}/probe_{arm}_{method}_d{dim}.json"

    params = arm_params(method, dim, arm)
    hidden = min(256, max(64, 32 * dim))
    rec = (json.load(open(out_path)) if os.path.exists(out_path) else
           {"method": method, "dim": dim, "arm": arm, "steps": rc.STEPS,
            "hparams": params, "seeds": []})
    done = {e["seed"] for e in rec["seeds"]}
    print(f"=== PROBE {arm} {method}_d{dim} done={len(done)}/{N_SEEDS} "
          f"[{sw.fmt(method, params)}]", flush=True)
    for s in range(N_SEEDS):
        if s in done:
            continue
        t0 = time.time()
        out = rc.run_seed(method, dim, s, params, hidden)
        rec["seeds"].append(out)
        json.dump(rec, open(out_path, "w"), indent=1)
        print(f"  seed {s:2d}: plateau={out['plateau']:8.3f} "
              f"closed={100*out['frac_closed']:5.1f}% err={out['error']} "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)
    fr = np.array([e["frac_closed"] for e in rec["seeds"]
                   if math.isfinite(e.get("frac_closed", float("nan")))])
    if len(fr):
        print(f"  PROBE CELL DONE frac_closed={100*fr.mean():.1f}%±"
              f"{100*fr.std(ddof=1)/math.sqrt(len(fr)):.1f} (n={len(fr)})",
              flush=True)


if __name__ == "__main__":
    main()
