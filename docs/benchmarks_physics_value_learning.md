# Physics / molecular benchmarks for value-function learning

Compiled 2026-07-18 from three parallel arXiv sweeps (Boltzmann/unnormalized-
density samplers; molecular generation; lattice/spin/transition-path/SOC).
All arXiv IDs in the shortlist were verified against the live arXiv API;
the molecular-slice agent had no web access, so its knowledge-based IDs were
re-verified by hand (all correct, with one relabel noted inline).

**Our setting.** We learn V(x,t) = log E[exp(r(X_1)) | X_t=x] under a fixed
base diffusion; e^r = e^{-energy} is a Boltzmann target. We want benchmarks
that are (1) small enough for one modest GPU, (2) let the base process be
fixed so only the value/control is learned, (3) have published baseline
numbers, (4) have an energy-potential reward. The neural-sampler literature
is a near-exact structural match: almost all of these methods learn a
control/drift u_θ(x,t) under a fixed forward process — which is ∇V up to a
known factor.

---

## Tier 0 — start here (closed-form energy, tiny nets, dense baselines)

### GMM-40 — 40-mode 2D Gaussian mixture
- 2D, closed-form energy (40 Gaussians, means in [-40,40]²). Multimodal;
  the canonical mode-coverage test. Reward r = −E is exactly the log-mixture.
- Baselines w/ numbers: PIS (2111.15141), DDS (2302.13834), DIS (2307.01198),
  iDEM (2402.06121), BNEM (2409.09787), Beyond-ELBOs (2406.07423), SCLD
  (2412.07081, uses a 50D variant). Metrics: W2/Sinkhorn to ground-truth
  samples, log-Z error, ELBO/EUBO, entropic mode coverage.
- Fixed base: yes (fixed VP/OU forward, learn drift). Energy-only, no dataset,
  trivial MLP, minutes on any GPU. **The obvious first target** — it's the
  same shape as our existing GMM dim-scaling family, so most of our harness
  ports directly and gives an external baseline to calibrate against.

### Funnel-10 — Neal's funnel
- 10D, closed-form, hard geometry (not multimodal). Baselines: PIS/DDS/DIS/
  Beyond-ELBOs/SCLD (log-Z, ELBO, ESS). Fixed base, energy-only, tiny MLP.
  A cheap geometry sanity target alongside GMM-40.

### Many-Well-32
- 32D = 16 coupled quartic double-wells → 2¹⁶ modes; the mode-coverage stress
  test with a clean Boltzmann reading. Baselines: FAB (2111.11510), DDS,
  Beyond-ELBOs, Kim-2025 scalable samplers (2505.19552, also 64/128D).
  Fixed base, energy-only. Closest to our high-d concern in closed form.

---

## Tier 1 — molecular n-body (the energy-potential targets you asked for)

### DW-4 (8D) and Lennard-Jones LJ-13 (39D) / LJ-55 (165D)
- SE(3)×Sₙ-invariant closed-form pairwise potentials (double-well; Lennard-
  Jones + harmonic confinement). e^{-E} Boltzmann. Genuine physics n-body.
- Baselines: iDEM (2402.06121, first to train LJ-55 from energy), BNEM
  (2409.09787), FAB, PITA (2506.16471). Metrics: energy histograms,
  interatomic-distance distributions, W2, ESS.
- Fixed base: yes; **energy-only, simulation-free** (a small dataset is used
  only for W2 *evaluation*). Net is a small EGNN (E(3)-equivariant GNN) —
  the one new architectural dependency vs our current MLP value net.
- Scale: DW-4 and LJ-13 comfortable on a modest GPU; **LJ-55 is the ceiling**
  — feasible but the expensive end. Reference code: github.com/jarridrb/DEM
  ships all four energies + EGNN nets.

### Alanine dipeptide (ALDP) — the realistic step-up
- 22 atoms (66D Cartesian, or internal coords). Reward = classical MD force
  field via OpenMM (not a toy closed form → heavier per-eval). Diagnostic:
  Ramachandran (φ,ψ) coverage + energy histograms + ESS.
- Baselines: FAB (2111.11510), SCLD (2412.07081), Variance-Tuned Diffusion
  (2505.21005), PITA (2506.16471). Fixed base feasible; single-GPU but needs
  the energy oracle in the loop. Reach for it after the closed-form targets.

---

## Tier 2 — the structural bullseye: learned object *is* a value/committor

These are worth calling out separately: the object they learn is literally
our V (a Doob h-transform log-potential) or the committor (a value function
solving backward Kolmogorov). Closest conceptual match, though baseline
density is thinner than the sampler suites.

### Doob's Lagrangian (Du et al. 2024, arXiv:2410.07974, NeurIPS 2024)
- Conditions a fixed Brownian motion with known drift toward rare endpoints
  via **Doob's h-transform — the h-function is exactly our V**. Base fixed;
  learns the trajectory potential. Benchmarks: Müller–Brown (2D) + alanine
  dipeptide. Sample-efficient variational objective, small, code released.
  **The single closest structural match in the whole sweep.**

### Neural committor benchmarks (Li–Lin–Ren 2019; Khoo–Lu–Ying 2019;
  Rotskoff–Vanden-Eijnden ~2021)
- The committor q(x)=P(hit B before A | x) is a value function solving a
  backward Kolmogorov PDE. Standard low-D tests: Müller–Brown, double-well;
  sub-100-dim, tiny nets. (These predate reliable arXiv-id lookup here —
  confirm IDs before citing.) Purest "learned object = value function" case,
  but the base is a fixed diffusion and the loss is variational (Dirichlet
  energy), not our exp-space regression — a different estimator on the same
  object, useful as a contrast.

### PIPS (Holdijk et al. 2022, arXiv:2207.02149, NeurIPS 2023)
- TPS ↔ Schrödinger bridge ↔ SOC with an NN policy; **MD base fixed, learn
  only the control** (= ∇V). Alanine dipeptide, polyproline, chignolin. Code
  released.

---

## Tier 3 — fixed-base sampler baselines & harnesses (numbers + code to reuse)

- **NETS** (Albergo & Vanden-Eijnden 2024, arXiv:2410.02711) — fixed base SDE,
  learn only a drift; benchmarks on GMM + **2D φ⁴ lattice field theory**; ESS
  numbers. The cleanest "fixed base + learned control + φ⁴ energy" paper.
- **PIS** (Zhang & Chen 2021, arXiv:2111.15141) — the template: fixed Brownian
  base, learned control via SOC. Funnel/GMM/many-well/LGCP.
- **Improved off-policy training of diffusion samplers** (Sendera et al. 2024,
  arXiv:2402.05098, NeurIPS 2024) — the go-to benchmarking paper unifying
  PIS/DDS/GFlowNet training; standard suite (25-GMM, funnel, manywell-32,
  LGCP) with published log-Z tables + code.
- **Beyond ELBOs** (Blessing et al. 2024, arXiv:2406.07423) — a drop-in
  benchmark harness: task suite × 16 methods × unified metrics (ELBO/EUBO,
  ESS, Δlog-Z, W2, MMD, entropic mode coverage).
  Code: github.com/DenisBless/variational_sampling_methods. Mostly
  synthetic/Bayesian (no LJ/ALDP), but the fastest path to apples-to-apples
  baseline tables.
- **φ⁴ flow-MCMC** (Albergo–Kanwar–Shanahan 2019, arXiv:1904.12072 + tutorial
  2101.08176) — canonical φ⁴ action/energy + metrics (ESS, integrated
  autocorrelation). Use for the target definition; NETS gives the fixed-base
  version.

---

## Molecular-generation route (fixed pretrained base, property reward)

Distinct from the samplers: reuse a small pretrained 3D-molecule diffusion
model frozen, learn guidance toward a property. Energy connection is partial
(HOMO-LUMO gap / strain energy are potential-like; dipole less so).
- **EDM** (Hoogeboom 2022, arXiv:2203.17003) / **GeoLDM** (Xu 2023,
  arXiv:2305.01140) — small EGNN bases on QM9 (~134k), public checkpoints;
  canonical per-property conditional-MAE baseline table. Classifier/value
  guidance on frozen EDM is exactly our setup.
- **Torsional Diffusion** (Jing 2022, arXiv:2206.01729) — tiny torus score
  model, public checkpoints, COV/AMR conformer baselines, plus an explicit
  torsional Boltzmann-generator (energy-reward) variant. Strong "fixable
  base" fit.
- **RTB / amortized posterior inference** (Venkatraman et al. 2024,
  arXiv:2405.20971) — fine-tunes a fixed prior diffusion toward a reward
  posterior; GFlowNet-style molecular rewards. Conceptually "value under a
  fixed base," but confirm a small released checkpoint before committing.
- Out of scope for a small machine: docking-reward SBDD (TargetDiff
  2303.03543 / DiffSBDD — CrossDocked + in-loop Vina), materials diffusion
  (MatterGen — ML-potential relaxation loops).

---

## Recommendation

**Fastest credible external result:** GMM-40 → Many-Well-32 → DW-4 → LJ-13,
in that order. GMM-40 and Many-Well reuse almost our entire GMM harness
(swap the reward for the published energy, fix a VP forward), give immediate
apples-to-apples numbers via the Beyond-ELBOs / DEM code, and Many-Well-32 is
the closed-form analogue of our high-d coverage-vs-concentration finding. DW-4
and LJ-13 add the genuine energy-potential story at the cost of one new
dependency (a small EGNN value net) and the DEM repo's energies.

**Highest-conceptual-payoff:** Doob's Lagrangian (2410.07974) on
Müller–Brown + alanine dipeptide — its learned object is literally our V via
the h-transform, so it's the natural place to claim a like-for-like win on
value-function *quality*, and it doubles as the entry to the real
molecular-chemistry (ALDP) setting.

**One dependency to weigh:** the molecular n-body targets need an EGNN value
network (E(3)/permutation equivariance) rather than our current MLP. GMM-40,
Many-Well, Funnel, Müller–Brown, and the committor/φ⁴ targets do **not** —
they run with the MLP value net we already have.
