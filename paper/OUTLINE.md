# Paper outline — the Spence loss for value-function learning in reward-guided diffusion

Working title: **The Spence Loss: Well-Conditioned Value-Function Learning for
Reward-Guided Diffusion**

> **Naming.** The paper calls it the **Spence loss** (reads well; avoids
> collision with "MSE/quadratic"). In the repo and experiment logs it is the
> **`quad`** loss (`losses/log_quadratic_bregman.py`) — same object. A footnote
> at first mention should note the dilogarithm/Spence-function origin.

## One-paragraph thesis

Reward-guided diffusion / diffusion fine-tuning is, via entropy-regularized
stochastic control, the problem of learning a value function
V(x,t) = log E_base[exp(r(X_1)) | X_t=x] — a conditional log-partition
function — whose gradient is the optimal control. Learning V by regression is
natural but the unbiased Monte-Carlo target lives in *exp* space (regressing in
log space is well-conditioned but biased by the Jensen gap, which is exactly the
signal we want), so the standard squared-error loss has gradients that
**explode** for over-predictions and, worse, **vanish** for under-predictions
(where the model is most wrong and clipping cannot help). We observe that any
Bregman divergence has the value function as its population minimizer, and that
such a loss acts on the optimizer only through a scalar **gradient weight**
w(u) = u·φ''(u) multiplying the exp-space residual — so the design space *is*
the choice of w. Squared error (w = 2u) and Itakura–Saito (w = 1/u) are its
opposite extremes. We introduce the **Spence loss**, obtained by *solving* for
the weight w(u) = ln u/(u−1) that makes the gradient **linear in the log-space
prediction on both sides**, with a closed-form numerically-stable gradient and
a value expressed through the Spence (dilogarithm) function. A secondary
contribution analyzes why high-dimensional value learning is signal-sparse
(log-E-exp is a soft-max dominated by rare, hard-to-find high-value targets)
and studies on-policy / importance-sampling schemes to combat it. We validate
on 2-moons, a carefully-designed dimensional-scaling benchmark, and the GMM-40
sampler benchmark, where our log-partition estimate is competitive with
published samplers.

## Contributions (as a list for the intro)

1. **A gradient-weight view of value learning.** Every Bregman divergence
   applied to the unbiased exp-space target has V as its population minimizer,
   and reduces to a single positive weight w(u) = u·φ''(u) on the residual
   e^p − e^q. This turns "pick a potential" into "pick a tail profile", and
   places squared error (w = 2u) and Itakura–Saito (w = 1/u) as the two
   extremes — the first inverting the second's asymmetry rather than fixing it.
2. **The Spence loss** — derived by solving for the weight w(u) = ln u/(u−1)
   whose gradient is linear in the log-space prediction in *both* tails, so
   neither over- nor under-predictions blow up or die. Closed-form gradient
   p(e^p−e^q)/(e^p−1), a numerically stable branch-wise implementation finite
   for all float inputs, and a value expressed via the Spence function.
3. **A signal-sparsity analysis** of high-dimensional value learning and an
   on-policy importance-sampling scheme that closes the gap to off-policy
   training across dimension 2–512, with a carefully-designed
   dimensional-scaling benchmark that yields clean statistical reads. (Scoped:
   the tuned on-policy recipe does not transfer unchanged to GMM-40; see §9.)
4. **Empirical validation** on 2-moons, dimensional scaling, and GMM-40
   (log-partition estimate competitive with / better than published samplers).

---

## Section-by-section plan (with mapping to our artifacts)

### 1. Introduction
- Reward-guided diffusion & fine-tuning matter; framed as stochastic control.
- The value function V(x,t)=log E[e^{r}|X_t=x] is the central object; optimal
  drift = base + 2a∇V (Doob h-transform / Feynman–Kac; e^{-V} ~ Boltzmann).
- The learning problem and the exp-space gradient pathology (teaser figure:
  gradient vs error for MSE/IS/Spence).
- Contributions list.

### 2. Setting: value functions for reward-guided diffusion — **drafted**
- Base diffusion dX = f dt + √(2a) dW, X_0=0, t: 0→1; terminal reward r;
  tilted path measure dP*/dP = e^{r(X_1)}/Z. Note the *sampler* time convention
  (0 = point mass → 1 = samples), not the DDPM denoising direction.
- **The base family is the pinned interpolant**, and this had to be pinned down
  from the code, not assumed: X_0=0, terminal marginal ν, Brownian bridge in
  between, induced drift f = E[(X_1−X_t)/(1−t) | X_t]. It is **zero iff
  ν = N(0,2aI)**. The dim-scaling problems use a GMM ν, so f ≠ 0 there —
  verified by simulating the SDE and matching its terminal law to `gmm_sample`
  (std 1.02–1.23 vs driftless BM's 1.414).
- Entropy-regularized control (Uehara et al. 2402.15194); KL = E∫|u|²/(4a)dt;
  soft value = log-partition; **optimal control u* = 2a∇V**; V(0,0) = log Z.
- The **H-martingale property** e^{V(x,t)} = E_base[e^{V(X_s,s)}|X_t=x], with
  two consequences that drive the paper: (i) targets are unbiased in *exp*
  space and only there; (ii) the identity is **twist-independent** via Girsanov
  — the proposal is a variance-reduction device, not a modeling choice. This is
  what licenses §6.
- ρ is the **Girsanov exponential over [t,s]**, not a single Euler increment:
  ρ = exp(−(1/2a)∫u·dB − (1/4a)∫‖u‖²dτ), an exponential martingale with unit
  mean under Novikov. The per-step factor −u·ΔB/(2a) − ‖u‖²Δt/(4a) is the
  Euler *discretization* of it (and is what the SMC code accumulates).
  Verified over 1/4/32 steps with an adapted, state-dependent, deliberately
  non-optimal twist.
- How targets are formed. **Split them into two kinds and keep the distinction
  visible:** *anchored* targets terminate at the true r and satisfy
  E[e^q|x_t] = e^{V(x_t,t)} exactly, no matter how wrong V_θ is; *bootstrapped*
  targets substitute V_θ at a later time and satisfy no such identity.
  - off-policy bridge anchors — x_1 must come from the **base marginal ν**;
    reward-aware anchoring would bias them.
  - **Cost caveat (do not drop):** anchors are cheap *per sample* (no
    trajectory integrated, one call to r each) but that is not cheap *in
    total* — being uninformative about where r is large, they can need many
    more reward evaluations to reach a given accuracy. When r is expensive (QM
    calculation, docking score, simulator rollout) the binding budget is total
    calls to r, and the ranking between constructions can invert. Our
    benchmarks have cheap analytic rewards, so we report sample counts;
    flagged as a limitation in §9.
  - on-policy bootstrapped SMC targets — q = log((1/k)Σ ρ_i e^{v_i}) over
    siblings of a resampled parent. **Define the backup operator explicitly**
    (T_s W)(x,t) = log E_base[e^{W(X_s,s)}|X_t=x]; then the *only* guarantee is
    E[e^q|x_t] = e^{(T_s V_θ)(x_t,t)} — unbiased in exp space for the backup of
    the current network and nothing else. Never write "unbiased target" without
    saying unbiased *for what*: since e^{V_θ} is not harmonic, the children's
    values need not estimate the true value at the parent, and error at time s
    is inherited, not corrected. What makes the scheme work is that the
    terminal step carries q = r(X_1) exactly and anchoring propagates
    backwards. V is a fixed point (T_s V = V) but we claim no convergence.
  - backward-noising expansion.
- **Non-obvious and worth keeping:** the backward kernel
  N((t/s)x_s, 2a·t(s−t)/s·I) **does not depend on ν** — conditioning a Brownian
  bridge on an interior point leaves a Brownian bridge — so the expansion trick
  works for base models with nontrivial drift, not just BM. Verified numerically
  against the GMM base (third moments agree to 1e−4).

> **Verification status.** All §2 identities were checked numerically on an
> analytic linear-reward instance: HJB residual 0; H-martingale at intermediate
> s and at s=1; Girsanov twist-independence under a deliberately non-optimal
> twist; backward-kernel marginal; and the KL formula. Re-run before submission
> if the conventions change.

> **Budget note.** §2 currently runs ~1.6pp against a ~1pp budget. Cut
> candidates, in order: the "base family" paragraph → App. C; eq (girsanov)'s
> specialization to the learned twist → inline; the three target constructions
> → compress to one paragraph + pointer, since §6 re-derives the expansion.

### 3. The problem with exp-space squared error — **drafted**
- Two parameterizations; the unbiased MC target is E[e^r] (exp space).
- Squared error D(e^q,e^p)=(e^q−e^p)²: gradient dL/dp = 2e^p(e^p−e^q)
  = 2e^{2p} − 2e^{p+q}. **Explodes** as p→+∞ (e^{2p}); **vanishes** as p→−∞
  (e^p→0) exactly when the model is most wrong. Vanishing is unrecoverable
  (clipping only bounds the explosion).
- **"Why not regress in log space?"** — the obvious escape, and it must be
  closed early or the reader stops here. (p−q)² is perfectly conditioned but
  minimized by E[q], not log E[e^q]; the Jensen gap ≈ ½Var(q|x_t) is largest
  precisely in the heavy-tailed regime, and *is* the soft-max signal. So the
  exp-space target is forced; only the divergence is negotiable.
- Empirical pathology: overflow, skipped batches, and "weight poisoning" (one
  non-finite prediction spike propagates) — our forensics from the loss work.
  The stabilized `exp_mse` is the baseline throughout, so comparisons isolate
  the divergence and not the arithmetic. Numbers forward-referenced to §5.
  [artifact: losses/exp_mse.py, the nonfinite-skip diagnostics]

### 4. Bregman divergences for value learning — **drafted**
- Definition D_φ(T,u)=φ(T)−φ(u)−φ'(u)(T−u). **Argument order matters**: the
  prediction goes in the *second* slot — that is the one whose minimizer is
  the plain mean. (Swapped, the minimizer is a quasi-arithmetic mean: the
  *harmonic* mean for φ=−ln u. Verified numerically; footnote in the paper.)
- **Key lemma (now one line):** ∂/∂p D_φ(e^q, e^p) = w(e^p)·(e^p − e^q) with
  w(u) := u·φ''(u) > 0. Linear in the target ⇒ population minimizer is
  log E[e^q] = V for *every* strictly convex φ. [App. D proof]
- **The design space is the weight w.** A Bregman loss reaches the optimizer
  only through w. Criteria: tail behavior of w·(residual) in p;
  scale-invariance; implied noise model.
  - **Squared error:** w = 2u — grows with the prediction ⇒ explodes above,
    dies below.
  - **Itakura–Saito** (φ=−ln u): w = 1/u ⇒ dL/dp = 1 − e^{q−p}. Saturates at
    +1 for over-prediction, **explodes** for under-prediction. It *inverts*
    squared error's asymmetry rather than removing it — and that is the point:
    it trades the unrecoverable failure (vanishing) for the clippable one.
    Scale-invariant; gamma/exponential noise model.
- **The Spence loss (ours):** *derived*, not asserted. Demand dL/dp ~ p as
  p→+∞ and ~ p·e^q as p→−∞; read off w(u) ~ ln u/u and w(u) ~ −ln u; the
  simplest smooth positive interpolant is **w(u) = ln u/(u−1)** (= reciprocal
  logarithmic mean of 1 and u; F''(u) = ln u/(u(u−1)); w(1)=1).
  - Closed-form gradient dL/dp = p(e^p−e^q)/(e^p−1); zero iff p=q (the
    apparent zero at p=0 is cancelled by the pole); minimizer = V.
  - **Both tails linear in the prediction** — state precisely: linear in p,
    *not* in the error p−q, and the under-prediction branch carries curvature
    e^q. Own that: it breaks scale-invariance and it up-weights exactly the
    rare high-value targets that dominate the log-sum-exp (ties to §6).
  - **Value via the Spence function** S(x)=Li₂(1−e^x): hence the name;
    monitored value only (gradient uses the closed form).
  - **Numerically stable branch-wise gradient** (expm1 forms) finite for all
    float inputs — fixes the naive-autograd NaN at p≳88.
    [artifact: losses/log_quadratic_bregman.py, algorithms/spence.py,
     tests/losses/ symbolic verification]
  - Implied observation model via Bregman↔exponential-family duality
    (Banerjee et al. 2005): φ* is not a standard named family, so we report
    the correspondence, not a closed-form likelihood. **Do not claim "gamma".**
- **Comparison table:** MSE / Itakura–Saito / Spence × {w(u), under-predict
  tail, over-predict tail, scale-inv, noise model}.

### 5. Off-policy experiments: the loss in isolation
- Controlled comparison Spence vs exp-MSE vs MSE at fixed setups.
- Metrics: training stability (fraction non-finite / skipped batches),
  accuracy (value RMSE vs analytic oracle; regret), and the **value-vs-gradient
  2×2** (extreme/incorrect × value/gradient) showing the gradient channel is
  where damage concentrates and where Spence wins.
  [artifact: bs4_moons + dim_scaling off-policy loss ablations]

### 6. Signal sparsity in high dimensions → on-policy learning (secondary)
- log E exp r is a soft-max: dominated by the highest-r targets, which are
  needle-in-haystack under base sampling; off-policy MC is inefficient in high d.
- Importance sampling / on-policy twisted SMC concentrates samples where the
  integrand is large; the learned V *is* the twist (self-consistent loop).
- **Best method (main body):** on-policy twisted SMC with a learned-value twist
  [name it]; brief statement of the coverage-vs-concentration nuance (at very
  high d, broad coverage beats corridor concentration; a backward-noising
  expansion restores small-t coverage).
- Alternatives (FBRRT/FBSDE, TD(λ), ancestral, guided proposals) + the
  hyperparameter-law single-family result: **Appendix B** with a comparison.

### 7. Experiments
- **7.1 Dimensional scaling** (headline for the sparsity/on-policy story):
  the constant-headroom, coordinate-nested, paired-seed design and *why* it
  gives clean statistical reads (dimension is the only varying factor;
  regret/headroom comparable across d). Results: loss comparison across d;
  on-policy vs off-policy crossover; the single law-fitted family matching or
  beating off-policy at every dimension.
  [artifact: dim_scaling_consth, dim_scaling_recipe, dim_scaling_lawv2]
- **7.2 2-moons:** value field + samples, Spence vs MSE, qualitative +
  quantitative. [artifact: bs4_moons]
- **7.3 GMM-40:** external Boltzmann benchmark; log-partition bias vs published
  samplers (iDEM 0.34, DDS 0.36, FAB 1.17, PIS 2.24 — ours 0.08), full mode
  coverage, and the **learned-V-vs-analytic-oracle** RMSE (a value-quality axis
  no baseline reports). [artifact: benchmarks_sampler]

### 8. Related work
- SOC / adjoint methods (Uehara 2402.15194; Domingo-Enrich Adjoint Matching
  2409.08861; SOCM 2312.02027). Log-variance / path-space losses
  (Nüsken–Richter 2005.05409). Diffusion samplers (PIS 2111.15141,
  DDS 2302.13834, DIS 2307.01198, FAB 2111.11510, iDEM 2402.06121,
  off-policy 2402.05098). Twisted SMC / FK steering (Zhao 2404.17546,
  TDS 2306.17775, FK-steering 2501.06848, FK-correctors 2503.02819,
  TRI-TSMC 2605.25123). Deep BSDE (1706.04702). Bregman divergences & proper
  scoring rules. GFlowNets.

### 9. Conclusion & limitations
- Spence loss as a drop-in fix for value-regression instability; on-policy
  helps in the sparse regime.
- Limitations: Many-Well-32 (value-representation failure — held out;
  App. or one honest paragraph); untuned transfer; molecular n-body (EGNN)
  and committor/φ⁴ targets as future work.
- **Reward-call budget.** All our benchmarks have cheap analytic rewards, so we
  measure cost in samples seen. Under an expensive r (QM, docking, simulator)
  the relevant budget is total evaluations of r, and the off-policy/on-policy
  ranking could differ. Say this plainly; §2.4 forward-references it.

---

## Appendices
- **A. The Spence loss:** derivation from the Bregman potential F, the
  dilogarithm identities (S'(x)=−x e^x/(e^x−1)=x/expm1(−x)), the
  population-minimizer proof, the branch-wise stable gradient, and the
  uncertainty/gamma interpretation. Symbolic (sympy) verification.
- **B. On-policy method zoo:** FBRRT/FBSDE (with the corrected backward
  targets), TD(λ) variants, ancestral resampling, gradient-based guided
  proposals; the coverage-vs-concentration probes; the hyperparameter-law
  fitting and the single-family (law-v2) result; per-method comparison tables.
- **C. Experimental details & reproducibility:** the constant-headroom
  calibration, nested-seed construction, network/optimizer, hyperparameter
  laws, compute.
- **D. Additional proofs:** H-martingale unbiasedness; Bregman minimizer;
  optimal-twist / zero-variance connection.

## Figures & tables (planned)
1. Fig 1: value + gradient vs prediction for MSE / IS / Spence at shared
   v_true ∈ {−2,0,2} — the two-sided linear-tail story. **Built**
   (`figures/gen_loss_curves.py`, imports the repo's real Spence value/gradient
   so figure and shipped loss cannot drift). Currently placed in §3; if §1
   needs a teaser, reuse the bottom row only.
2. Fig 2: 2-moons value field & samples, Spence vs MSE.
3. Table 1: off-policy loss ablation — stability + accuracy + 2×2 damage.
4. Fig 3: dimensional scaling — frac_closed vs d (loss ablation; on- vs
   off-policy; single-family curve).
5. Table 2 / Fig 4: GMM-40 — log-Z bias vs published, mode coverage,
   learned-V-vs-oracle RMSE.
6. (App) on-policy zoo comparison; hyperparameter laws; Spence derivation
   figures.

## Locked decisions
- **Venue & length:** ML conference, ~9pp main + appendix. Budget guide:
  §1 ~0.75pp, §2 ~1pp, §3 ~0.75pp, §4 (loss) ~2pp, §5 ~1.25pp, §6 ~1pp,
  §7 ~1.75pp, §8+§9 ~0.5pp. Everything below "keep tight" moves to appendix.
- **On-policy scope: sparsity + best method only.** §6 states the
  signal-sparsity argument and the single best on-policy method; §7.1 shows the
  dim-scaling crossover. **To Appendix B:** the recipe (expansion + finer
  integration), the hyperparameter-law single-family (law-v2) result, the
  coverage-vs-concentration probes, the FBRRT/TD-λ/ancestral/guided-proposal
  zoo, and the trust-region probe. The recipe's non-transfer to GMM-40 is an
  honest appendix note, not a main-body result.
- **Name:** "Spence loss" (repo `quad`), per the naming box above.

## Drafting order (recommended)
1. §4 Bregman/Spence — **drafted** (`sections/04_bregman.tex`).
2. §3 exp-MSE pathology + Fig 1 — **drafted** (`sections/03_expmse.tex`).
3. §2 Setting (H-martingale targets) — **drafted** (`sections/02_setting.tex`).
4. App. D proofs + App. A Spence derivation. → **next up.** App. A must carry:
   the φ/φ'' derivation from w, the dilogarithm identities, the branch-wise
   gradient, and the w-sandwiching remark (Spence lies between the MSE and IS
   weights *asymptotically* — the global sandwich fails in a small band near
   u≈0.6, so do not overstate). App. D: H-martingale unbiasedness, the Bregman
   minimizer (now a two-line consequence of the weight form), and the
   Girsanov/twist-independence statement.
5. §5 off-policy loss ablation (Table 1) — pull numbers from bs4_moons/dim_scaling.
6. §7 experiments (dim-scaling Fig 3, 2-moons Fig 2, GMM-40 Table 2).
7. §6 sparsity + best method; App. B zoo.
8. §1 intro + §8 related + §9 conclusion last, once the story is fixed.

# Building the PDF

No system LaTeX; use the self-contained `tectonic` engine:

    tectonic -X compile main.tex --outdir build

(one-time: download the static binary from the tectonic GitHub releases).
