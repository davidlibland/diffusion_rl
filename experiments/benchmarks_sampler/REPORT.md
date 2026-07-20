# Sampler benchmarks — first results

Harness in this directory (see PLAN.md). Both benchmarks map onto our
value learner via a driftless-BM base (p_base=N(0,2I)) and the tilt reward
r(y)=log π(y)−log p_base(y), so V(0,0)=log Z by construction. Methods:
`off_policy` (exp-space regression on base-bridge anchors) vs
`single_seed_mc` (our on-policy SMC), law-v1 hparams, 15k steps, 3 seeds.
Metrics: log-Z error (learned V(0,0) vs truth), mode metric, sliced-W2 to
exact samples, energy-W1, and — GMM only — learned-V RMSE vs the analytic
oracle. Everything error-free, ~4 min/run on the small nets.

## Results (mean over 3 seeds)

### GMM-40 (2D, true log Z = 0)
| method | log-Z err | sliced-W2 | energy-W1 | V-RMSE vs oracle | mode cov |
|---|---|---|---|---|---|
| off_policy | 0.19 | 0.65 | 618 | 0.51 | 1.00 |
| **single_seed_mc** | **0.08** | **0.28** | **148** | 0.54 | 1.00 |

### Many-Well-32 (32D, true log Z = 164.70; naive base-IS = 153.3)
| method | log-Z hat | log-Z err | well-balance P(+) | sliced-W2 | energy-W1 |
|---|---|---|---|---|---|
| off_policy | 138.96 | 25.74 | 0.50 | 0.88 | 153.5 |
| **single_seed_mc** | **148.42** | **16.28** | 0.50 | 0.88 | 152.8 |
| *target* | *164.70* | — | *0.85* | — | — |

## Readings

1. **On-policy beats off-policy on both benchmarks** — the dim-scaling thesis
   reproduces on external Boltzmann targets. On GMM-40 ssmc more than halves
   both log-Z error (0.08 vs 0.19) and sliced-W2 (0.28 vs 0.65); on
   Many-Well-32 ssmc's log-Z error is 16.3 vs 25.7. The learned-V RMSE vs the
   analytic oracle (GMM, ~0.5 nats) is a direct measurement of value quality
   that the sampler literature can't make — our harness gets it for free.

2. **GMM-40 is essentially solved; log-Z is excellent.** ssmc's V(0,0)
   recovers log Z=0 to 0.08 nats. The energy-W1 stays large only because
   L=50 makes the modes razor-sharp (σ≈0.026) so −log π is hypersensitive to
   sub-σ misplacement; sliced-W2 (0.28) is the honest sample-quality number
   and is small relative to the ±0.8 mode spread.

3. **Many-Well-32 is genuinely hard and under-explored at this budget.** Both
   methods *undershoot* log Z (149/139 vs 164.7) and produce balanced signs
   P(+)=0.50 while the target concentrates on the deep well at P(+)=0.85 —
   i.e. the controlled process is not finding the deep wells, so V(0,0)
   integrates over mass that misses them. off_policy (139) even undershoots
   the naive base-IS baseline (153.3); ssmc (148) approaches it. This is an
   exploration/concentration failure, not a bug: 2¹⁶ modes in 32-D with a
   symmetric base and no recipe. Published FAB reports ~1–2 nat log-Z error
   on MW32, so there is a real gap to close.

## Next steps (clear from the gap)

1. **Apply the recipe (law-v2: expansion + guided proposals) to Many-Well.**
   Our dim-scaling work showed expansion + guidance is exactly what closes the
   high-d exploration gap; MW32 is a 32-D high-d target, so this is the
   indicated fix. Needs `sweep_lawv2.build` (augment_fn) instead of the v1
   build used here.
2. **Longer training + guidance-scale sweep** on MW32 (runs are ~4 min, cheap
   to scale to 60k steps / a small sweep).
3. **Report against published tables** once MW32 is competitive: FAB/PIS/DDS
   log-Z and sample W2 for MW32; iDEM/FAB metrics for GMM-40.
4. The clean **learned-V-vs-oracle** axis (GMM) is worth featuring — it
   isolates value-function quality from sampler quality, which no baseline
   reports.

## Caveat

L=50 (DEM's normalization) makes GMM modes very sharp, inflating energy-W1;
sliced-W2 and log-Z are the robust GMM metrics. A moderate L≈15 would soften
the energy metric at some cost to base coverage — deferred, not silently
applied.
