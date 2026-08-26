# 24-hour paper iteration plan

## The loop (revised from the user's proposal)

0. **Inventory + claim ledger** — every claim in the paper gets a row in
   `CLAIMS.md`: statement, where it appears, evidence type (proof / numerical
   check / experiment artifact), status. New claims may not enter the paper
   without a row.
1. **Freeze the numbers.** Extract a canonical `results.json` from the Optuna
   DBs and per-seed grid JSONs before drafting §5/§7, so rewrites cannot
   drift numbers.
2. **Close evidential gaps with experiments** (ranked by claim-importance x
   support-weakness x cost), launched in background FIRST so writing overlaps
   compute.
3. **Write** — I draft anything claim-critical myself. Agents gather and
   review only.
4. **Evaluate with fresh context** — reviewers see only the compiled PDF plus
   a rubric; the author context is blind to its own gaps.
5. **Fix, rebuild, push.** Return to 2.

## Why these changes

The dominant failure mode in this project has been *confidently asserting
false things*: swapped Bregman argument order, reversed Itakura-Saito tails,
assuming a driftless base, single-step Girsanov ratio, "unbiased for the
backup". None of those would be caught by prose review, so the ledger and the
numeric checks are the core of the loop, not an add-on.

## Style target (decided once, not re-litigated)

ML-conference *structure* (contributions list, signposting, empirical claims
forward) with mathematical *precision* in S2/S4 (stated lemmas, explicit
operators, no hand-waving asymptotics). Li-Bland's math papers supply the
precision register: terse, low hedging, defined-before-used notation, results
announced flatly rather than sold. They do NOT supply the structure -- a
def-thm-proof spine would be wrong for this venue.

## Stopping criteria

- All sections + appendices drafted.
- Every CLAIMS.md row is `verified` or explicitly hedged in the text.
- Fresh-context reviewers raise no `major` findings.
- Clean build, no unresolved refs, figures regenerate from scripts.

## THE BIG GAP (found during inventory)

The paper's headline claim -- the Spence loss beats exp-space MSE -- is
currently supported by:
  - `experiments/loss_comparison/`: 2D moons, quad vs mse, ONE seed, no saved
    numbers (only a PNG);
  - incidental evidence from bs4_moons sweeps where `loss` was a tuned
    categorical (fbrrt_cv -> quad, fbrrt -> quad, fbrrt_td_lambda -> mse).

That is not enough for the central claim of the paper. The dim-scaling
studies all pin `LOSS = "quad"` and never ablate it.

**Priority experiment 1 (`experiments/loss_ablation/`).** Controlled,
multi-seed, multi-dimension loss ablation on the constant-headroom nested GMM
benchmark, off-policy only (isolates the loss from the sampler):
  - arms: `quad` (Spence), `mse` (exp-MSE), `is` (Itakura-Saito),
    `logmse` (log-space squared error -- the biased baseline of S3);
  - **per-loss lr tuning** before the seed grid, so the comparison is
    tuned-vs-tuned (otherwise a reviewer correctly objects that the shared
    hyperparameters were tuned under `quad`);
  - metrics: frac_closed (headroom closed), value RMSE vs the analytic
    oracle, and non-finite/skipped-batch counts (the stability axis of S3);
  - paired nested seeds, so per-dimension deltas are paired tests.

This single experiment supplies Table 1, the S4 three-way comparison as
*empirical* rather than purely theoretical, and the S3 Jensen-bias claim.
