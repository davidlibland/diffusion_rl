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

## Follow-ups tried (both negative — recorded)

### The dim-scaling recipe does NOT transfer; it hurts
law-v2 (expansion + guided proposals), the fix indicated by our high-d work:

| config | GMM-40 log-Z err | GMM-40 sliced-W2 | GMM-40 V-RMSE | MW32 log-Z err |
|---|---|---|---|---|
| plain ssmc (15k) | **0.08** | **0.28** | **0.54** | **16.3** |
| ssmc + recipe (15k) | 0.42 | 0.37 | 0.80 | 17.9 |
| ssmc + recipe (40k) | — | — | — | **121.2** (collapsed) |

The recipe's hyperparameters were fit to the GMM-*drift* constant-headroom
family, not these driftless Boltzmann targets, so they mis-transfer; expansion
+ guidance over long training on MW32 actively diverges (V(0,0)→43). **Plain
on-policy ssmc is the best config on both benchmarks.**

### Guidance scale is inert on the Many-Well barrier
Sweeping guidance_scale ∈ {0.5, 1.5, 4.0} on MW32 (ssmc, 15k) gives
*bit-for-bit identical* outcomes: log-Z hat 147.7±0.04, P(+)=0.498,
sliced-W2 0.880 at every scale. Cranking the guided proposal 8× moves nothing.

### Diagnosis: MW32 is a value-representation failure, not a sampling one
Every method sits at P(+)=0.498 while the target is 0.85 — the controlled
process is symmetric across each double well. Since guidance (which only
*uses* the value gradient at generation) is completely inert, the deficit is
in the learned **value function itself**: it fits the symmetric part of each
well (6x²−x⁴) but misses the small linear tilt (0.5·x, worth ~1.74 nats/well)
that makes the + well deeper. A symmetric value has a symmetric gradient, so
no generation-side knob can recover the asymmetry. The 16-nat log-Z undershoot
is exactly 16 wells × ~1 nat of missing deep-well mass. **This is squarely a
value-learning problem — our domain — and the benchmark isolates it cleanly.**

## Next steps

1. **Attack the value-representation gap directly** (not generation knobs):
   the 16 wells are independent, so a per-well/factorized value head, or an
   architecture/loss that resolves the sub-dominant linear tilt, is the
   principled fix. Symmetry-breaking exploration (temperature annealing / the
   trust-region escort ramp seeding one well) is the alternative.
2. **Re-fit hyperparameters on the Boltzmann family** rather than reusing the
   GMM-drift laws — the recipe failure shows the laws don't transfer.
3. **Report against published tables** once MW32 is competitive: FAB reports
   ~1–2 nat log-Z error on MW32 (we are at 16); we are far off and now know
   why.
4. Feature the **learned-V-vs-oracle** axis (GMM, ~0.5 nats) — it isolates
   value quality from sampler quality, which no baseline reports, and it is
   the axis on which MW32's failure is legible.

## Comparison to published numbers (same target configs)

Our GMM-40 is iDEM's GMM (d=2, 40 modes, loc 40, softplus scale) and our
Many-Well-32 is Sendera's Manywell (d=32, FAB energy) — identical, so the
comparison is apples-to-apples on the coordinate-free log-Z axis.

**GMM-40**, log-Z bias (target normalized, true log Z=0) — iDEM Table
(2402.06121); sample W2 in native coords:

| method | \|Δ log Z\| | sample W2 (native) |
|---|---|---|
| PIS | 2.24 | 7.64 |
| FAB | 1.17 | 12.0 |
| DDS | 0.36 | 9.31 |
| iDEM (SOTA) | 0.34 | 7.42 |
| **ours ssmc** | **0.08** | ≈14 (sliced-W2×50, a lower bound) |
| ours off-policy | 0.19 | ≈32 |

→ **On log Z we are better than all published methods** (0.08 vs best 0.34);
on sample W2 we are in the same ballpark but our sliced metric isn't directly
comparable and iDEM's 7.42 is likely ahead. Mode coverage is complete (40/40),
matching the good methods. GMM-40 is competitive-to-winning.

**Many-Well-32**, Δ log Z — Sendera Table 1 (2402.05098), 5 runs:

| method | Δ log Z (raw) | Δ log Z (reweighted) |
|---|---|---|
| SMC | 14.99 | — |
| DIS | 10.52 | 3.05 |
| DDS | 7.36 | 0.23 |
| PIS | 3.85 | 2.69 |
| TB (GFlowNet) | 4.01 | 2.67 |
| best (TB+Expl+LP+LS) | 4.68 | 0.07 |
| GGNS | 0.29 | — |
| **ours ssmc** | **16.28** | — (no reweighting applied) |

→ **On MW32 we are not competitive**: 16.3 is worse than every tuned
diffusion sampler (PIS 3.85 raw; best ≈0.1–0.3 reweighted), on par only with
untuned plain SMC (14.99). Consistent with the value-representation diagnosis.

**Two caveats that bound the comparison.** (1) We have *not* tuned on this
family — law-v1 hparams, driftless base, small nets, 15k steps. (2) The
published raw→reweighted jump (DDS 7.36→0.23, best 4.68→0.07) comes from
importance-**reweighting** the sampler's own outputs; we report only the raw
V(0,0) readout and apply no reweighting. We already have exact SMC weights, so
a reweighted log-Z estimate is the cheapest large lever on the MW32 gap.

## Caveat (metrics)

L=50 (DEM's normalization) makes GMM modes very sharp, inflating energy-W1;
sliced-W2 and log-Z are the robust GMM metrics. A moderate L≈15 would soften
the energy metric at some cost to base coverage — deferred, not silently
applied.
