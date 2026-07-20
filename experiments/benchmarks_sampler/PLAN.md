# Sampler-benchmark evaluation plan

Goal: evaluate our value-function learner V(x,t)=log E[exp(r(X_1))|X_t=x] on
established **physics/Boltzmann sampling benchmarks** where the reward is an
energy (e^{-E} is the target density), the base diffusion is **fixed**, and
published baselines exist. This document is the plan; `problem_sampler.py`
implements the targets and `run_benchmark.py` runs them.

## How a Boltzmann benchmark maps onto our framework

A sampler benchmark specifies an unnormalized target π(x) ∝ e^{-E(x)}. We fix
the base process as **driftless Brownian motion** from X_0=0 (the PIS/DIS/DEM
reference), dX = √(2a) dW on t∈[0,1], so the base terminal is
p_base = N(0, 2a·I). With a=1 (matching the rest of the codebase) that is
N(0, 2I). The optimal controlled terminal in our framework is
p_base(x)·e^{r(x)}/Z; setting that ∝ e^{-E} gives the **tilt-correction
reward**

    r(x) = −E(x) − log p_base(x) + const = −E(x) + ‖x‖²/(4a) + const,

and the optimal control u = 2a·∇V steers the fixed Brownian reference to
sample π. So the benchmark becomes exactly our value-learning problem with a
driftless base and this reward. Coordinates are normalized by a per-problem
factor L (targets with far-flung modes are divided by L so they sit at O(1)
scale inside the N(0,2I) reference support, matching DEM's
`data_normalization_factor`); the energy is evaluated at L·y.

This is a **cleaner** fit than our GMM-drift problem family: the off-policy
regressor's Brownian-bridge anchors x_t = t·x_1 + √(2a t(1−t))·ε are exactly
correct here (the base really is driftless BM), which our on_policy code
flagged as the precondition for `off_policy_frac`.

Fixed base + published baselines + energy reward: all satisfied. What we vary
and learn is only V / the control.

## Metrics (matching the literature)

- **log-Z error** |log Ẑ − log Z|: log Z = V(0,0) in our notation (the base
  process starts at 0), so our learned V(0,0) *is* a normalizing-constant
  estimate — directly comparable to the ELBO/log-Z tables in PIS/DDS/FAB/iDEM.
  Ground-truth log Z is known in closed form for Many-Well and estimable to
  high accuracy for GMM.
- **Mode coverage**: fraction of the target's modes with a generated sample
  nearby (GMM-40: 40 modes; Many-Well-32: 2¹⁶ sign-pattern modes → fraction of
  distinct sign patterns hit).
- **Energy W1 / histogram distance**: 1-Wasserstein between the energy
  distribution of generated vs ground-truth samples (the standard DEM/iDEM
  diagnostic).
- **Sample W2** (GMM-2D, if POT available): 2-Wasserstein to exact samples.
- **Oracle V check** (GMM only): the target is a Gaussian mixture and the base
  is Gaussian, so V(y,t) is analytic (Gaussian algebra). We validate the
  learned V against it on a grid — a luxury the real benchmarks lack, and a
  direct measurement of *value-function quality* rather than only sample
  quality.

## The two benchmarks built now

### GMM-40 — 40-mode 2D Gaussian mixture
- **Definition (FAB / DEM `GMM`, arXiv:2402.06121, 2111.11510):** dim=2,
  n_mixes=40, means ~ Uniform[−40,40]² (torch seed 0), equal weights,
  covariance = softplus(1.0)²·I ≈ 1.72·I. Energy E(x) = −log Σ_k (1/40)
  N(x; μ_k, 1.72·I). Normalization L=50 (DEM's factor).
- **Why:** the canonical mode-coverage sampler benchmark; nearly our entire
  GMM harness ports over. Dense baselines (PIS, DDS, DIS, iDEM, BNEM,
  Beyond-ELBOs). Has an analytic V for validation.
- **Ground truth:** exact samples (sample the mixture); log Z computable to
  high precision; 40 known mode locations.

### Many-Well-32 — 16 coupled double wells
- **Definition (FAB `ManyWellEnergy`, arXiv:2111.11510):** dim=32, n_wells=16.
  Each 2D well: energy = (−0.5·x₁ − 6·x₁² + x₁⁴) + 0.5·x₂², i.e. a quartic
  double-well in the odd coordinate (modes ≈ ±1.7) and a unit Gaussian in the
  even coordinate. E(x) = Σ over the 16 wells. 2¹⁶ modes. No normalization
  (L=1); already O(1). Unnormalized target; **log Z known in closed form**
  (per-well 2D log-Z × 16).
- **Why:** the closed-form high-dimensional mode-coverage stress test — the
  direct analogue of our coverage-vs-concentration finding, in a physics
  Boltzmann target. Baselines: FAB, DDS, Beyond-ELBOs, Kim-2025 (2505.19552).
- **Ground truth:** exact samples (per-well rejection sampling for the
  double-well dim + Gaussian for the other); closed-form log Z; 2¹⁶ sign-mode
  structure for coverage.

## References (arXiv IDs verified against the live API)

| tag | paper | id | uses |
|---|---|---|---|
| PIS | Path Integral Sampler (Zhang & Chen 2021) | 2111.15141 | fixed-base template; GMM/funnel/manywell |
| FAB | Flow Annealed IS Bootstrap (Midgley et al. 2021) | 2111.11510 | GMM-40, Many-Well-32 definitions + numbers |
| DDS | Denoising Diffusion Sampler (Vargas 2023) | 2302.13834 | GMM/funnel/manywell baselines |
| DIS | Improved sampling via learned diffusions (Richter & Berner 2023) | 2307.01198 | log-variance loss; same suite |
| iDEM | Iterated Denoising Energy Matching (Akhound-Sadegh 2024) | 2402.06121 | GMM-40 config; energy-matching baseline |
| BNEM | Bootstrapped Noised Energy Matching (OuYang 2024) | 2409.09787 | GMM/DW4/LJ energy baselines |
| Beyond-ELBOs | Large-scale eval of variational sampling (Blessing 2024) | 2406.07423 | benchmark harness + metrics |
| Off-policy samplers | Improved off-policy training (Sendera 2024) | 2402.05098 | manywell/funnel/GMM log-Z tables |
| Scalable samplers | Kim et al. 2025 | 2505.19552 | Manywell 32/64/128 |
| NETS | Non-Equilibrium Transport Sampler (Albergo & VE 2024) | 2410.02711 | fixed-base + φ⁴ (next tier) |
| Doob's Lagrangian | Du et al. 2024 (h-transform = our V) | 2410.07974 | conceptual bullseye (next tier) |

Full survey and the wider shortlist (DW-4/LJ-13/LJ-55, alanine dipeptide,
committor and φ⁴ targets): `../../docs/benchmarks_physics_value_learning.md`.

## Sequence

1. **GMM-40** — validate learned V vs analytic oracle; report log-Z error,
   mode coverage, energy-W1. (This file's runner.)
2. **Many-Well-32** — log-Z error, mode coverage, energy-W1 vs published FAB
   numbers. Same runner, `--problem manywell`.
3. Compare off-policy regression vs our best on-policy recipe (law-v2 config
   adapted to the driftless base) on both, mirroring the dim-scaling story.
4. (Later) DW-4 / LJ-13 (needs an EGNN value net) and Doob's-Lagrangian
   Müller–Brown / alanine dipeptide.
