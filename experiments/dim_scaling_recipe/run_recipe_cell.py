#!/usr/bin/env python3
"""One (arm, method, dimension) cell of the RECIPE dimension-scaling grid.

The dim_scaling_consth study found on-policy SMC methods lose to off-policy
regression at d >= 256 and traced the deficit to (a) EM integrator bias at
the law's n_steps=19 and (b) thin training coverage at small t.  Probe 3
fixed both: n_steps=60 plus BACKWARD-NOISING EXPANSION -- every on-policy row
(x_s, s, v-hat) is expanded to a sample at smaller t through the exact base
backward kernel X_t | X_s = x_s ~ N((t/s) x_s, 2a t(s-t)/s I) (the base
process is an h-transform of BM from X_0=0, so its past given the present is
a Brownian bridge from 0), reusing v-hat as the exp-space regression target
(consistent by the harmonic property e^{V(x,t)} = E_base[e^{V(X_s,s)}|X_t=x]
for base-marginal sources).  At d=512 this put ssmc and ssmc-td ABOVE
off-policy on paired seeds 0-9.

This grid runs the recipe over ALL dims {2..512} x 30 nested seeds to plot
the full comparison against off-policy.  Methodology is identical to
../dim_scaling_consth/run_cell.py (constant 6-nat headroom, coordinate-nested
paired instances, law-fitted hparams served by the FROZEN fit_consth copy in
this directory, 15k steps, plateau = tail mean of smoothed val reward).

Arms:
  expand_ns60      : law hparams + n_steps=60 + expansion (k=1 per row).
  expand_ns60_sub  : same, then each epoch subsampled uniformly back to
                     DS_BATCH x (that method's law n_steps) rows, restoring
                     the law control's dataset regeneration cadence
                     (freshness-matched variant; best at d=512 for ssmc).

Controls (off_policy, ssmc, ssmc-td law configs) are NOT re-run: the
../dim_scaling_consth/results grid used the identical protocol on the same
nested instances/seeds and is read directly by plot_recipe.py.

Results: results/recipe_<arm>_<method>_d<dim>.json (per-seed, resumable).
Usage:   python run_recipe_cell.py --method single_seed_mc --dim 64 \
             --arm expand_ns60_sub
Env:     DSR_N_SEEDS=30 DSC_STEPS=15000 DSC_VAL_EVERY=500 DSC_TAIL=8
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
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback

HERE = os.path.dirname(os.path.abspath(__file__))
# Frozen local copies (sweep_consth, fit_consth, problem_consth) shadow the
# originals; dim_scaling_bs4 supplies the shared base modules.
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "dim_scaling_bs4"))
sys.path.insert(0, HERE)

import sweep_consth as sw  # noqa: E402  (frozen copy)
from fit_consth import hparams_for_dim  # noqa: E402  (frozen copy)
from problem_consth import make_problem  # noqa: E402  (frozen copy)
from problem import optimal_terminal_and_reward  # noqa: E402

RESULTS = os.environ.get("DSR_RESULTS_DIR", os.path.join(HERE, "results"))
N_SEEDS = int(os.environ.get("DSR_N_SEEDS", 30))
STEPS = int(os.environ.get("DSC_STEPS", 15000))
VAL_EVERY = int(os.environ.get("DSC_VAL_EVERY", 500))
TAIL = int(os.environ.get("DSC_TAIL", 8))

T_MIN = 0.05        # don't expand below this t (kernel variance -> 0 anyway)


def make_expand_fn(a, t_min=T_MIN, k=1):
    """Backward-noising augmentation (see module docstring)."""
    def expand(all_x, all_t, all_tgt, all_w):
        src = torch.isfinite(all_tgt) & (all_t > t_min)
        if not src.any():
            return all_x, all_t, all_tgt, all_w
        xs = all_x[src].repeat(k, 1)
        ss = all_t[src].repeat(k)
        ys = all_tgt[src].repeat(k)
        ws = all_w[src].repeat(k)
        tp = t_min + torch.rand_like(ss) * (ss - t_min)
        var = (2.0 * a * tp * (ss - tp) / ss).clamp_min(0.0)
        xp = ((tp / ss).unsqueeze(-1) * xs
              + torch.sqrt(var).unsqueeze(-1) * torch.randn_like(xs))
        return (torch.cat([all_x, xp]), torch.cat([all_t, tp]),
                torch.cat([all_tgt, ys]), torch.cat([all_w, ws.clone()]))

    return expand


def with_subsample(fn, n_rows):
    """Wrap an augment_fn with uniform row subsampling to n_rows, restoring
    the law control's epoch size / regeneration cadence."""
    def inner(all_x, all_t, all_tgt, all_w):
        all_x, all_t, all_tgt, all_w = fn(all_x, all_t, all_tgt, all_w)
        n = all_x.shape[0]
        if n > n_rows:
            idx = torch.randperm(n, device=all_x.device)[:n_rows]
            all_x, all_t, all_tgt, all_w = (all_x[idx], all_t[idx],
                                            all_tgt[idx], all_w[idx])
        return all_x, all_t, all_tgt, all_w

    return inner


class ValCollector(Callback):
    def __init__(self):
        super().__init__(); self.vals = []

    def on_validation_end(self, trainer, pl):
        m = trainer.callback_metrics.get("val_reward_mean")
        if m is not None:
            self.vals.append(float(m))


def run_seed(method, dim, s, arm, hidden):
    prob = make_problem(dim, seed=s)
    _, e_opt, _ = optimal_terminal_and_reward(
        prob["means"], prob["sigma2"], prob["weights"], prob["c"],
        prob["reward_scale"], dim)
    params = hparams_for_dim(method, dim)
    law_ns = int(params.get("n_steps", 19))
    params["n_steps"] = 60

    model, vm, ds, loader = sw.build(method, params, prob, dim, hidden, s)
    ds.augment_fn = make_expand_fn(ds.a)
    if arm == "expand_ns60_sub":
        ds.augment_fn = with_subsample(ds.augment_fn, sw.DS_BATCH * law_ns)

    vc = ValCollector()
    tr = L.Trainer(max_steps=STEPS, val_check_interval=VAL_EVERY,
                   callbacks=[vc], logger=False, enable_checkpointing=False,
                   enable_progress_bar=False, num_sanity_val_steps=0)
    err = None
    try:
        tr.fit(model, loader, val_dataloaders=sw.val_loader)
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
    gc.collect(); sw.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--arm", required=True,
                    choices=["expand_ns60", "expand_ns60_sub"])
    args = ap.parse_args()
    method, dim, arm = args.method, args.dim, args.arm
    os.makedirs(RESULTS, exist_ok=True)
    out_path = f"{RESULTS}/recipe_{arm}_{method}_d{dim}.json"
    hidden = min(256, max(64, 32 * dim))
    rec = (json.load(open(out_path)) if os.path.exists(out_path) else
           {"method": method, "dim": dim, "arm": arm, "steps": STEPS,
            "seeds": []})
    done = {e["seed"] for e in rec["seeds"]}
    print(f"=== RECIPE {arm} {method}_d{dim} done={len(done)}/{N_SEEDS}",
          flush=True)
    for s in range(N_SEEDS):
        if s in done:
            continue
        t0 = time.time()
        out = run_seed(method, dim, s, arm, hidden)
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
