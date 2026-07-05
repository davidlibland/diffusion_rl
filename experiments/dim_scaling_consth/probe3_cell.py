#!/usr/bin/env python3
"""Round-3 probe: close the coverage-vs-concentration loop at d=512.

Round 2 established that concentration alone HURTS at high d: giving ssmc the
oracle twist (perfect corridor concentration, per-step ESS 0.64-0.90) scored
16.3% closed vs 25.9% for the law's do-nothing twist (ESS=1.00) and 30.7% for
off-policy.  Off-policy's whole advantage is broad base-marginal coverage.
Round 3 tests whether corridor data helps ON TOP of coverage, and implements
the noising-expansion idea: an on-policy trajectory carries value estimates
v-hat all along its length; each (x_s, s, v-hat) can be EXPANDED backward to
smaller t (higher noise) through the exact base backward kernel

    X_t | X_s = x_s  ~  N((t/s) x_s,  2a t(s-t)/s I),      t < s,

(the base process is an h-transform of BM from X_0=0, so its past given the
present is a Brownian bridge from 0 -- x_1 cancels and the kernel is exact,
not approximate).  By the harmonic property e^{V(x,t)} = E_base[e^{V(X_s,s)}
| X_t=x], the pair (x', t') with target v-hat(x_s, s) is a consistent
exp-space regression sample PROVIDED the sources x_s are base-marginal at
time s.  Corridor sources (twisted chain, marginal ~ p_s e^{V}) must be
un-weighted by w ~ e^{-v-hat} (self-normalized) to restore that condition.

Arms (ssmc, d=512, paired seeds, 15k steps):

  blend         : oracle twist + off_policy_frac=0.5.  Coverage from the
                  off-policy splice, concentration from the twist.  If this
                  beats BOTH parents (16.3 / 30.7), corridor data adds signal
                  once coverage is guaranteed; if it matches off-policy, the
                  corridor adds nothing.
  expand        : law config + backward-noising augmentation (k=1 extra
                  sample per source row, no unweighting -- the law twist has
                  ESS=1.00 so trajectory marginals ARE base marginals).
                  Densifies the small-t region with propagated terminal
                  signal; tests the mechanism in its realizable form.
  expand_oracle : oracle twist + augmentation WITH e^{-v-hat} unweighting.
                  The mechanism at full strength: corridor sources carry the
                  most informative v-hat, expansion + unweighting spreads
                  them over the base manifold at smaller t.

Controls (existing): grid ssmc d512 25.9%, probe2 oracle_twist 16.3%,
grid off_policy d512 30.7% (seeds 0-9 means).

Usage: python probe3_cell.py --method single_seed_mc --dim 512 --arm expand
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

RESULTS = os.environ.get("DSC_RESULTS_DIR", os.path.join(HERE, "results"))
N_SEEDS = int(os.environ.get("DSC_PROBE_SEEDS", 10))

T_MIN = 0.05        # don't expand below this t (kernel variance -> 0 anyway)
W_MAX = 100.0       # cap on any single unweighted row after normalization


def make_expand_fn(a, unweight, t_min=T_MIN, k=1):
    """Backward-noising augmentation for OnPolicySMCDataset.augment_fn.

    For every source row (x_s, s, y, w) with s > t_min, draws k samples
    t' ~ U(t_min, s), x' ~ N((t'/s) x_s, 2a t'(s-t')/s I) and appends
    (x', t', y, w') to the epoch.  With unweight=True the appended rows get
    w' = w * e^{-y} (self-normalized to mean 1), correcting corridor-marginal
    sources back to the base marginal required by the harmonic property.
    """
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
        if unweight:
            lw = torch.nan_to_num(-ys, nan=-torch.inf)
            lw = lw - lw.max()
            wp = ws * torch.exp(lw)
            tot = wp.sum()
            if tot > 0:
                wp = (wp * (wp.numel() / tot)).clamp(max=W_MAX)
            else:
                wp = torch.ones_like(wp)
        else:
            wp = ws.clone()
        return (torch.cat([all_x, xp]), torch.cat([all_t, tp]),
                torch.cat([all_tgt, ys]), torch.cat([all_w, wp]))

    return expand


def run_seed_arm(method, dim, s, arm, hidden):
    prob = make_problem(dim, seed=s)
    _, e_opt, _ = optimal_terminal_and_reward(
        prob["means"], prob["sigma2"], prob["weights"], prob["c"],
        prob["reward_scale"], dim)
    params = hparams_for_dim(method, dim)
    if arm == "blend":
        params["off_policy_frac"] = 0.5

    model, vm, ds, loader = sw.build(method, params, prob, dim, hidden, s)
    if arm in ("blend", "expand_oracle"):
        ds.smc_value = prob["anal_fn"]                     # tau = V*
    if arm in ("expand", "expand_oracle"):
        ds.augment_fn = make_expand_fn(ds.a, unweight=(arm == "expand_oracle"))

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
    del model, vm, ds, loader, tr
    import gc
    gc.collect(); sw.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--arm", required=True,
                    choices=["blend", "expand", "expand_oracle"])
    args = ap.parse_args()
    method, dim, arm = args.method, args.dim, args.arm
    os.makedirs(RESULTS, exist_ok=True)
    out_path = f"{RESULTS}/probe3_{arm}_{method}_d{dim}.json"
    hidden = min(256, max(64, 32 * dim))
    rec = (json.load(open(out_path)) if os.path.exists(out_path) else
           {"method": method, "dim": dim, "arm": arm, "steps": rc.STEPS,
            "seeds": []})
    done = {e["seed"] for e in rec["seeds"]}
    print(f"=== PROBE3 {arm} {method}_d{dim} done={len(done)}/{N_SEEDS}",
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
