#!/usr/bin/env python3
"""Round-2 failure-mode probe: integrator bias + the ssmc value lock-in.

Round 1 (probe_cell.py) ruled out twist strength, particle count (9->16), and
the gradient driver at the tested magnitudes.  Round 2 tests the two remaining
mechanisms at d=512, paired seeds 0-9, full 15k steps:

  ns60          (ssmc, ssmc-td, amctl): law config with n_steps 19/22 -> 60.
                The EM integrator at 19 steps biases the on-policy sampling
                distribution by -0.86 nats of terminal reward at d=512 (14% of
                the prize); off-policy trains on exact samples.  Mechanical fix.

  oracle_twist  (ssmc): the twist is the ANALYTIC value function (tau = V*).
                Audit: with the exact twist, per-step ESS drops to ~0.9/0.64
                and the twisted chain's terminals cover ~half the base->optimal
                reward distance by resampling alone -- vs ESS=1.00 (zero
                concentration) under the law's learned twist.  Tests the
                LOCK-IN hypothesis: ssmc's targets are twist-telescoped MC
                estimates, unbiased for any twist but with variance set by
                twist quality; an uninformative twist (untrained V) leaves raw
                MC of an e^{-gap}-rare expectation -> no signal -> V never
                improves -> twist never improves.

  oracle_guid   (ssmc): guidance = grad V* at scale 1 (law twist kept).
                Separates concentration-by-resampling from
                transport-by-drift as the escape route.

  warmstart     (ssmc): stage 1 trains OFF-POLICY for 15k steps; stage 2 runs
                ssmc with twist AND guidance from the frozen stage-1 network.
                The realizable version of the oracle arms -- and the direct
                test of "can ssmc leverage a decent V to OUTPERFORM
                off-policy", since stage-2 ssmc gets exactly the V that
                off-policy ends with.

Usage: python probe2_cell.py --method single_seed_mc --dim 512 --arm oracle_twist
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import lightning as L

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "dim_scaling_bs4"))
sys.path.insert(0, HERE)

import run_cell as rc  # noqa: E402
import sweep_consth as sw  # noqa: E402
from fit_consth import hparams_for_dim  # noqa: E402
from problem_consth import make_problem  # noqa: E402
from problem import optimal_terminal_and_reward  # noqa: E402
from diffusion_rl.models.on_policy import grad_value_guidance  # noqa: E402

RESULTS = os.environ.get("DSC_RESULTS_DIR", os.path.join(HERE, "results"))
N_SEEDS = int(os.environ.get("DSC_PROBE_SEEDS", 10))


def run_seed_arm(method, dim, s, arm, hidden):
    prob = make_problem(dim, seed=s)
    _, e_opt, _ = optimal_terminal_and_reward(
        prob["means"], prob["sigma2"], prob["weights"], prob["c"],
        prob["reward_scale"], dim)
    params = hparams_for_dim(method, dim)

    warm_vm = None
    if arm == "ns60":
        params["n_steps"] = 60
    elif arm == "warmstart":
        # stage 1: off-policy for the same budget; its final V seeds stage 2.
        off_params = hparams_for_dim("off_policy", dim)
        m0, warm_vm, ds0, ld0 = sw.build("off_policy", off_params, prob, dim,
                                         hidden, s)
        tr0 = L.Trainer(max_steps=rc.STEPS, limit_val_batches=0, logger=False,
                        enable_checkpointing=False, enable_progress_bar=False,
                        num_sanity_val_steps=0)
        tr0.fit(m0, ld0)
        # Lightning returns the module to CPU on fit() teardown; re-pin it.
        warm_vm = m0.value_module.to(sw.DEVICE).eval()
        for p_ in warm_vm.parameters():
            p_.requires_grad_(False)
        del m0, ds0, ld0, tr0

    model, vm, ds, loader = sw.build(method, params, prob, dim, hidden, s)
    if arm == "oracle_twist":
        ds.smc_value = prob["anal_fn"]                     # tau = V*
    elif arm == "oracle_guid":
        ds._guidance = grad_value_guidance(prob["anal_fn"], 1.0)
    elif arm == "warmstart":
        wv = warm_vm

        def warm_fn(x, t):
            return wv(x, t.reshape(-1))

        ds.smc_value = warm_fn                             # tau = V_off
        ds._guidance = grad_value_guidance(warm_fn, 1.0)   # u = grad V_off

    vc = rc.ValCollector()
    tr = L.Trainer(max_steps=rc.STEPS, val_check_interval=rc.VAL_EVERY,
                   callbacks=[vc], logger=False, enable_checkpointing=False,
                   enable_progress_bar=False, num_sanity_val_steps=0)
    err = None
    try:
        tr.fit(model, loader, val_dataloaders=sw.val_loader)
    except (RuntimeError, ValueError) as e:
        err = f"{type(e).__name__}: {str(e)[:80]}"
    v = np.array(vc.vals, dtype=float)
    import pandas as pd
    if len(v):
        sm = pd.Series(v).rolling(rc.TAIL, min_periods=1).mean().to_numpy()
        plateau = float(np.nanmean(sm[-rc.TAIL:])); best = float(np.nanmax(sm))
    else:
        plateau = best = float("nan")
    out = {"seed": s, "plateau": plateau, "best": best,
           "opt_reward": float(e_opt), "E_base": prob["diag"]["E_base_r"],
           "regret": plateau - float(e_opt),
           "frac_closed": ((plateau - prob["diag"]["E_base_r"]) / 6.0
                           if math.isfinite(plateau) else float("nan")),
           "nonfinite_skips": int(getattr(model, "_nonfinite_count_total", 0)),
           "error": err}
    del model, vm, ds, loader, tr, warm_vm
    import gc
    gc.collect(); sw.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--arm", required=True,
                    choices=["ns60", "oracle_twist", "oracle_guid", "warmstart"])
    args = ap.parse_args()
    method, dim, arm = args.method, args.dim, args.arm
    os.makedirs(RESULTS, exist_ok=True)
    out_path = f"{RESULTS}/probe2_{arm}_{method}_d{dim}.json"
    hidden = min(256, max(64, 32 * dim))
    rec = (json.load(open(out_path)) if os.path.exists(out_path) else
           {"method": method, "dim": dim, "arm": arm, "steps": rc.STEPS,
            "seeds": []})
    done = {e["seed"] for e in rec["seeds"]}
    print(f"=== PROBE2 {arm} {method}_d{dim} done={len(done)}/{N_SEEDS}",
          flush=True)
    for s in range(N_SEEDS):
        if s in done:
            continue
        t0 = time.time()
        out = run_seed_arm(method, dim, s, arm, hidden)
        rec["seeds"].append(out)
        json.dump(rec, open(out_path, "w"), indent=1)
        print(f"  seed {s:2d}: plateau={out['plateau']:8.3f} "
              f"closed={100*out['frac_closed']:5.1f}% err={out['error']} "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)
    fr = np.array([e["frac_closed"] for e in rec["seeds"]
                   if math.isfinite(e.get("frac_closed", float("nan")))])
    if len(fr):
        print(f"  DONE frac_closed={100*fr.mean():.1f}%±"
              f"{100*fr.std(ddof=1)/math.sqrt(len(fr)):.1f}", flush=True)


if __name__ == "__main__":
    main()
