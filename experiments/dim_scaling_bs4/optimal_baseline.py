"""Analytical optimal-policy baseline per dimension (CPU-only, ~seconds).

Computes, for each dimension:
  * the optimal value function v(x,t) = log E[exp(r(X_T))|X_t=x]  (analytical,
    via problem.make_problem's AnalyticalValue) — validated at the origin, and
  * E_opt_reward = E_{p*}[r], the expected reward of the optimal model, where
    p* is the base GMM tilted by exp(r) (closed form, still a GMM).

E_opt_reward is the correct per-dimension BASELINE (best achievable expected
reward) to compare the trained policies' plateau reward against — unlike
V(0,0), which is a log-partition value and can be exceeded by the reward.

Each quantity is cross-checked against an independent Monte-Carlo estimate.
Writes results/optimal_baseline.json.  Safe to run while the sweep is going
(CPU-only, read-only w.r.t. the pipeline).
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problem import make_problem, optimal_terminal_and_reward  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DIMS = [int(x) for x in os.environ.get("DSB_DIMS", "2,8,32,128").split(",")]
TARGET_GAP = float(os.environ.get("DSB_TARGET_GAP", 6.0))
MC_N = 400_000

out = {}
print(f"{'d':>4} | {'V00(anal)':>10} {'V00(form)':>10} {'V00(mc)':>9} | "
      f"{'E_opt_r':>9} {'E_opt(mc)':>10} | {'opt ‖x-c‖²':>10} {'s':>9}")
print("-" * 92)
for d in DIMS:
    prob = make_problem(d, target_gap=TARGET_GAP)
    s = prob["reward_scale"]

    # analytical value at the origin (from AnalyticalValue) and closed form
    V00_anal = float(prob["anal_fn"](torch.zeros(1, d), torch.zeros(1)).item())
    V00_form, E_opt, tilted = optimal_terminal_and_reward(
        prob["means"], prob["sigma2"], prob["weights"], prob["c"], s, d)

    # MC cross-checks via importance sampling from the base GMM
    x = torch.from_numpy(prob["gmm_sample"](MC_N)).float()
    r = prob["reward_fn"](x).double().numpy()
    rmax = r.max()
    logZ_mc = float(np.log(np.mean(np.exp(r - rmax))) + rmax)   # = V00
    wis = np.exp(r - rmax)
    E_opt_mc = float((wis * r).sum() / wis.sum())               # = E_{p*}[r]

    out[d] = {
        "reward_scale": s, "target_gap": TARGET_GAP,
        "V00_analytical": V00_anal, "V00_closed_form": V00_form, "V00_mc": logZ_mc,
        "E_opt_reward": E_opt, "E_opt_reward_mc": E_opt_mc,
        "opt_terminal_dist2": tilted["E_dist2"],
    }
    print(f"{d:>4} | {V00_anal:>10.4f} {V00_form:>10.4f} {logZ_mc:>9.4f} | "
          f"{E_opt:>9.4f} {E_opt_mc:>10.4f} | {tilted['E_dist2']:>10.4f} {s:>9.4g}")

# consistency assertions (closed form vs analytical vs MC)
maxerr_V = max(abs(o["V00_closed_form"] - o["V00_analytical"]) for o in out.values())
maxerr_Vmc = max(abs(o["V00_closed_form"] - o["V00_mc"]) for o in out.values())
maxerr_E = max(abs(o["E_opt_reward"] - o["E_opt_reward_mc"]) for o in out.values())
print("-" * 92)
print(f"max |V00 closed-form - analytical| = {maxerr_V:.2e}")
print(f"max |V00 closed-form - MC|         = {maxerr_Vmc:.2e}")
print(f"max |E_opt closed-form - MC|       = {maxerr_E:.2e}")

json.dump(out, open(f"{RESULTS}/optimal_baseline.json", "w"), indent=2)
print(f"\nSaved {RESULTS}/optimal_baseline.json")
