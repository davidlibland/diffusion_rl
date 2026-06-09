"""Analytical regret of the UNTRAINED BASE model (no guidance) per dimension.

The base model samples X_1 ~ p_base, so its expected reward is the exact
E_base[r] = -s * sum_k w_k (||mu_k - c||^2 + d*sigma_k^2).  We report
regret_base = E_base[r] - E_{p*}[r]  (same sign convention as the method
regrets; <= 0 since the optimal tilt can only raise the mean reward).  This is
the "do nothing" reference: |regret_base| = E_opt - E_base is the control
headroom the methods are competing for.

Runs for BOTH experiments (gap-calibrated multiseed and matched), prints the
regret tables augmented with a `base` column, and saves base_regret.json in
each results dir.  CPU-only, analytical, non-destructive.
"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BS4 = os.path.join(ROOT, "dim_scaling_bs4")
MATCHED = os.path.join(ROOT, "dim_scaling_matched")
sys.path.insert(0, BS4)
from problem import optimal_terminal_and_reward  # noqa: E402


def e_base_r(prob, d):
    mu = np.asarray(prob["means"]); s2 = np.asarray(prob["sigma2"])
    w = np.asarray(prob["weights"]); c = np.asarray(prob["c"]); s = prob["reward_scale"]
    d2 = ((mu - c[None, :]) ** 2).sum(1) + d * s2          # E[||x-c||^2] per comp
    return float(-s * (w * d2).sum())


def base_regret_for(make_problem, dims, n_seeds):
    out = {}
    for d in dims:
        regs = []
        for s in range(n_seeds):
            prob = make_problem(d, seed=s)
            _, e_opt, _ = optimal_terminal_and_reward(
                prob["means"], prob["sigma2"], prob["weights"], prob["c"],
                prob["reward_scale"], d)
            regs.append(e_base_r(prob, d) - e_opt)
        a = np.array(regs)
        out[d] = {"mean": float(a.mean()),
                  "sem": float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0,
                  "n": len(a)}
    return out


METHODS = ["off_policy", "single_seed_mc", "single_seed_td_lambda", "ancestral_mc_td_lambda"]
SHORT = {"off_policy": "off-policy", "single_seed_mc": "ssmc",
         "single_seed_td_lambda": "ssmc-td(λ)", "ancestral_mc_td_lambda": "anc-mc-td(λ)"}


def report(label, results_dir, make_problem, n_seeds):
    summ = json.load(open(f"{results_dir}/summary.json"))
    dims = summ["dims"]
    base = base_regret_for(make_problem, dims, n_seeds)
    json.dump(base, open(f"{results_dir}/base_regret.json", "w"), indent=2)

    print(f"\n{'='*104}\n{label}: regret = plateau − optimal (mean ± SEM); "
          f"`base` = untrained base model (analytical)\n{'='*104}")
    hdr = f"{'dim':>5} | " + " | ".join(f"{SHORT[m]:>17}" for m in METHODS) + " | " + f"{'base (untwisted)':>18}"
    print(hdr); print("-" * len(hdr))
    for d in dims:
        cells = []
        for m in METHODS:
            r = summ["regret"][m].get(str(d))
            cells.append(f"{r['mean']:>10.3f} ±{r['sem']:.2f}" if r else f"{'—':>17}")
        b = base[d]
        cells.append(f"{b['mean']:>11.3f} ±{b['sem']:.2f}")
        print(f"{d:>5} | " + " | ".join(cells))
    print(f"saved {results_dir}/base_regret.json")


# gap-calibrated multiseed
from problem import make_problem as make_gap  # noqa: E402
report("GAP-CALIBRATED (−V(0,0)=6), 30 seeds",
       os.path.join(ROOT, "dim_scaling_multiseed", "results"), make_gap, 30)

# matched calibration
sys.path.insert(0, MATCHED)
from problem_matched import make_problem as make_matched  # noqa: E402
report("MATCHED (E_base[r]=−10), 10 seeds",
       os.path.join(ROOT, "dim_scaling_matched", "results"), make_matched, 10)
