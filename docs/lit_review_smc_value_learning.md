# Literature review: SMC / guidance for accelerating value-function learning

Compiled 2026-07-16 from three parallel arXiv sweeps (twisted-SMC/diffusion
alignment; SOC/soft-value/FBSDE; RL/inference-time search). WebSearch was
unavailable to the survey agents, so everything was verified against arXiv
abstract pages directly; 2026 arXiv IDs (26xx/25xx.*) should be spot-checked
before citing externally.

Setting: we learn V(x,t) = log E[exp(r(X_1)) | X_t=x] under a base diffusion;
SMC with a learned twist generates on-policy training data for exp-space value
regression. Established internally: coverage beats concentration at high d;
backward-noising expansion of on-policy value estimates + finer integration
lets on-policy SMC value learning beat off-policy regression (dim_scaling_*).

---

## 1. Closest neighbors: the "SMC data → value learning" loop

**TRI-TSMC — Trust-Region Iterative Twisted SMC** (Wang et al., 2026,
arXiv:2605.25123). Iteratively learns diffusion twists inside SMC: each
iteration is a KL-constrained (trust-region) path-space update with a
closed-form tempered-importance-reweighting solution, projected back onto the
twist family by weighted maximum likelihood. Proves the optimal twist gives a
zero-variance sampler; tempered path provably shrinks residual weight
variance. *Key insight:* bound the per-iteration distribution shift (trust
region) instead of freezing targets — a KL trust region as the
staleness/target-network mechanism, and an explicit answer to "which
training-state distribution": the current tempered particle measure.
**The single most relevant paper to our loop.**

**Twisted SMC for LMs / Contrastive Twist Learning** (Zhao et al., ICML 2024,
arXiv:2404.17546). Origin of learned twists-as-values (twist = expected future
terminal potential), soft-RL connection, contrastive twist learning from
positive/negative samples. *Key insight:* bidirectional SMC bounds on log Z
give a two-sided KL diagnostic — effectively a principled coverage-vs-
concentration meter for twist quality (we currently use ESS, which is one-sided).

**CDM — Contrastive Distribution Matching** (Kim et al., 2026,
arXiv:2605.23346). Amortizes the optimal twist for discrete diffusion via a
contrastive objective exploiting closed-form forward kernels; <5% inference
overhead. *Key insight:* closed-form forward kernels → variance-reduced twist
gradients; the "amortize SMC away" direction (opposite of ours, useful foil).

**Self-Distilled Twisted SMC** (Kim, 2025, arXiv:2507.02315). For sparse
rewards, iteratively distill the base model toward the target so twist
estimation gets easier. *Key insight:* move the proposal, not just the
weights, when reward signal is too sparse for twist learning to start.

**Gap / positioning:** none of these frames SMC-with-learned-twist explicitly
as *on-policy training-data generation for value regression that beats
off-policy regression*, nor reports a coverage-beats-concentration result at
high dimension, nor uses backward-noising expansion of value estimates. Those
three appear to be our differentiators.

---

## 2. SOC / adjoint family: avoid the value function (or stabilize its targets)

**Adjoint Matching** (Domingo-Enrich et al., ICLR 2025, arXiv:2409.08861).
KL-regularized fine-tuning = SOC; *memoryless* noise schedule removes the
initial-condition bias; regress the control onto a backward lean-adjoint ODE —
**no value function appears at all**. Companion taxonomy (arXiv:2410.00345)
shows it dominates alternatives on gradient variance.

**RAM — Reinforce Adjoint Matching** (Bergmeister et al., 2026,
arXiv:2605.10759). The optimal KL-regularized process only tilts the clean
endpoint law; so train on **on-policy endpoints re-noised through the original
forward process** with reward-weighted denoising regression — no controlled
SDE rollouts. Up to 50× fewer steps than Flow-GRPO. *Key insight:* re-noising
on-policy endpoints as the training distribution — **structurally the same
move as our backward-noising expansion** (theirs re-noises endpoints; ours
re-noises intermediate value estimates via the exact bridge kernel).

**Unified fine-tuning/sampling perspective** (Domingo-Enrich, Du, Albergo,
2026, arXiv:2605.00229). Proves adjoint matching (and "novel score matching")
have **finite gradient variance while target/conditional score matching do
not** — the sharpest published which-targets-blow-up-in-high-d result; plus
Crooks/Jarzynski identities for exponential tilts (our e^V martingale
machinery in thermodynamic form).

**Time-reversed BSDEs** (Mei & Taghvaei, 2026, arXiv:2603.20455). The lean
adjoint is not adapted to the forward filtration (peeks at the future) →
high-variance gradients; a time-reversed BSDE adjoint is adapted. *Key
insight:* adaptedness of the target as a martingale-structure variance
criterion — directly relevant to our right-endpoint FBRRT driver choice.

**Log-variance losses** (Nüsken & Richter, arXiv:2005.05409; follow-ups by
Richter & Berner). Loss = variance of the pathwise log-RN derivative under an
*arbitrary* reference measure; log Z cancels inside the variance; gradient
variance vanishes at the optimum ("sticking the landing"). *Key insight:* an
off-policy-valid log-space loss family with built-in partition-function
cancellation — a candidate alternative to our loss_shift trick.

**SOCM** (Domingo-Enrich et al., NeurIPS 2024, arXiv:2312.02027). Importance-
weighted matching loss valid under any sampling control, with reparameterization
matrices *learned to minimize the variance of the regression target itself*.
*Key insight:* treat target variance as an optimizable quantity.

**Q-learning + adjoint hybrids** (QAM, Li & Levine, arXiv:2601.14234; TRQAM,
arXiv:2605.27079; MaxEnt-RL diffusion policies, arXiv:2606.22630 — all 2026).
Bellman/TD critics with replay married to adjoint matching for the policy
update. The field's two answers to bootstrap instability: EMA/target nets
(this branch) vs eliminating V entirely (adjoint branch).

**Value Gradient Sampler** (Hwang et al., AISTATS 2026, arXiv:2502.13280).
Particles moved along ∇V with V trained by TD **with off-policy replay** —
direct precedent for TD-based soft-value learning in diffusion-style samplers.

**Adjoint Sampling** (Havens et al., 2025, arXiv:2504.11713). Replay buffer +
reciprocal projection → more gradient updates than energy evaluations —
concrete replay recipe for sample-efficient control/value learning.

---

## 3. Fixed-potential FK steering (inference-only; what a learned V upgrades)

- **FK Steering** (Singhal et al., 2025, arXiv:2501.06848) — the canonical
  taxonomy of heuristic potentials + intermediate resampling; 0.8B steered >
  2.6B fine-tuned.
- **Feynman-Kac Correctors** (Skreta et al., ICML 2025, arXiv:2503.02819) —
  exact analytic FK weights for annealed/product/tilted targets; the FK weight
  enforces the E[e^{V_{t+dt}}|x_t] = e^{V_t} consistency *by reweighting
  instead of regression*. Discrete version: Hasan et al., 2026,
  arXiv:2601.10403.
- **TDS** (Wu et al., NeurIPS 2023, arXiv:2306.17775) — origin: heuristic
  Tweedie twist, asymptotically exact with any twist quality.
- **SVDD** (Li et al., 2024, arXiv:2408.08252) — soft-value decoding with two
  estimators; **SVDD-MC regresses r (α→0 limit) on base-process states
  explicitly because exp-space targets are numerically unstable** — published
  confirmation of our quad-loss/loss_shift problem, solved by retreating to
  log space rather than stabilizing exp space as we did.

---

## 4. RL / reasoning side

- **Q♯** (Zhou et al., 2025, arXiv:2502.20548) — distributional RL for the
  optimal KL-regularized Q, provably convergent, guides a reference policy;
  variance-dependent convergence bounds.
- **Value-Guided Search** (Wang et al., 2025, arXiv:2505.17373) — 1.5B
  token-level value model trained on 2.5M traces (dense targets, huge
  coverage) instead of concentrated step-level PRM labels — a deliberately
  coverage-first value-training design that works.
- **Reject, Resample, Repeat** (Golowich et al., 2026, arXiv:2603.07887) —
  theory: the quantities controlling SMC sampling error do NOT control final
  task accuracy — faithful sampling of the tilted target ≠ better outcomes.
  **The formal cousin of our concentration-can-hurt result.**
- **PG-DLM** (Dang et al., 2025, arXiv:2507.08390) — particle-Gibbs
  (conditional SMC, keep-the-reference-trajectory) over whole denoising
  trajectories; a low-variance resampling mechanism we don't currently use.
- **Piché et al.** (ICLR 2019) — the classic ancestor: SAC-style soft value
  supplies SMC planning weights.

---

## 5. Actionable shortlist for this codebase

1. **Trust-region twist updates (TRI-TSMC):** replace/augment our EMA-decay
   knob with an explicit KL bound between successive twist-implied path
   measures (closed-form via tempering). Candidate cure for the residual
   fbrrt/bootstrap staleness sensitivity and a principled version of the
   epoch_rows cadence knob.
2. **Bidirectional log-Z bounds (Zhao et al.):** add as a training-time
   diagnostic; two-sided KL replaces one-sided ESS for detecting lock-in vs
   over-concentration.
3. **Adaptedness-aware targets (Mei & Taghvaei):** check our FBRRT
   right-endpoint fused driver against their time-reversed BSDE — same
   variance argument; may explain the remaining fbrrt high-d gap (still
   unaudited).
4. **Log-variance loss (Nüsken-Richter):** a partition-free log-space loss
   valid off-policy; natural A/B against quad+loss_shift, especially at
   d=512 where loss_shift was load-bearing.
5. **Conditional SMC / particle Gibbs (PG-DLM):** retain a reference
   trajectory across epochs — variance reduction orthogonal to everything we
   ship today.
6. **RAM's re-noised-endpoint training distribution:** our expansion already
   generalizes this; their 50× efficiency claim suggests pushing expansion
   fractions/depth further, and citing RAM as the nearest published analogue.

## 6. Where our results sit

Published work either (a) learns twists from SMC data but for *inference-time
alignment*, evaluating sample quality (TRI-TSMC, CDM, Zhao), or (b) learns
values/controls for fine-tuning while deliberately *avoiding* the value
(adjoint family), or (c) steers with fixed potentials (FK family). The
specific claims we can support that none of these make: (i) on-policy
SMC-generated data measurably beats off-policy regression for *value learning
itself* at high dimension, once integrator bias and small-t coverage are
fixed; (ii) concentration (even oracle-twisted) *hurts* value learning at high
d — with Golowich et al. 2026 as the theoretical cousin; (iii) exact
bridge-kernel backward-noising expansion of intermediate value estimates as a
coverage restorer, with the harmonic property as its correctness argument.
