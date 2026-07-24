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
natural but the unbiased Monte-Carlo target lives in *exp* space, so the
standard squared-error loss has gradients that **explode** for over-predictions
and, worse, **vanish** for under-predictions (where the model is most wrong and
clipping cannot help). We observe that any Bregman divergence has the value
function as its population minimizer, freeing us to choose the generating
potential for conditioning rather than convenience. We introduce the **Spence
loss**, the Bregman divergence whose gradient has **quadratic tails on both
sides** — bounded curvature, non-vanishing under-prediction signal — with a
closed-form numerically-stable gradient and a value expressed through the
Spence (dilogarithm) function. It sits in a family with squared error and the
Itakura–Saito divergence as reference points. A secondary contribution
analyzes why high-dimensional value learning is signal-sparse (log-E-exp is a
soft-max dominated by rare, hard-to-find high-value targets) and studies
on-policy / importance-sampling schemes to combat it. We validate on 2-moons,
a carefully-designed dimensional-scaling benchmark, and the GMM-40 sampler
benchmark, where our log-partition estimate is competitive with published
samplers.

## Contributions (as a list for the intro)

1. **The Spence loss** — a Bregman divergence tailored to learning the
   conditional log-partition / value function, with quadratic gradient tails
   on both sides, a closed-form stabilized gradient, and a Spence-function
   value; the first loss to fix *both* the exploding- and (critically) the
   vanishing-gradient pathologies of exp-space regression.
2. **A Bregman-divergence view of value learning**, placing squared error and
   the Itakura–Saito divergence as endpoints of a design space and giving
   criteria (tail behavior, scale-invariance, uncertainty model) for choosing
   the potential.
3. **A signal-sparsity analysis** of high-dimensional value learning and an
   on-policy importance-sampling scheme, with a carefully-designed
   dimensional-scaling benchmark that yields clean statistical reads.
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

### 2. Setting: value functions for reward-guided diffusion
- Base diffusion dX = f dt + √(2a) dW; terminal reward r; target ∝ p_base·e^r.
- Entropy-regularized control (Uehara et al. 2402.15194 as the anchor);
  soft value = log-partition; optimal control = ∇V.
- The **H-martingale property** e^{V(X_t)} = E[e^{V(X_{t+dt})}|X_t]: gives
  twist-independent *unbiased* regression targets (the object we regress onto).
- How targets are formed: bridge sampling (x_t), terminal reward or bootstrapped
  value as q. (Off-policy vs on-policy foreshadowed.)

### 3. The problem with exp-space squared error
- Two parameterizations; the unbiased MC target is E[e^r] (exp space).
- Squared error D(e^p,e^q)=(e^p−e^q)²: gradient dL/dp = 2e^p(e^p−e^q)
  = 2e^{2p} − 2e^{p+q}. **Explodes** as p→+∞ (e^{2p}); **vanishes** as p→−∞
  (e^p→0) exactly when the model is most wrong. Vanishing is unrecoverable
  (clipping only bounds the explosion).
- Empirical pathology: overflow, skipped batches, and "weight poisoning" (one
  non-finite prediction spike propagates) — our forensics from the loss work.
  [artifact: losses/exp_mse.py, the nonfinite-skip diagnostics]

### 4. Bregman divergences for value learning
- Definition D_φ(p,q)=φ(p)−φ(q)−⟨∇φ(q),p−q⟩; squared error = quadratic φ.
- **Key lemma:** for any strictly convex φ, the population minimizer of
  E_q[D_φ(u, e^q)] over u is E[e^q]; so p=log u=log E[e^q]=V. *Every* Bregman
  divergence is a proper loss for the value — choose φ for conditioning.
  [App. A proof]
- Design criteria: gradient tail behavior in the natural variable p;
  scale-invariance; implied noise/uncertainty model.
- **Itakura–Saito** (φ=−ln u): linear tail resolves vanishing on one side,
  still explodes on the other; **scale-invariant**; corresponds to a
  **gamma / exponential** noise model (uncertainty in V). A useful midpoint.
- **The Spence loss (ours):** Bregman divergence with F''(u) = −ln u/(u(1−u)).
  - Closed-form gradient dL/dp = p(e^p−e^q)/(e^p−1); zero iff p=q; population
    minimizer = V.
  - **Quadratic tails both sides:** p→+∞ ⇒ dL/dp→p (no e^{2p} blow-up);
    p→−∞ ⇒ dL/dp→p·e^q (non-vanishing corrective signal).
  - **Value via the Spence function** S(x)=Li₂(1−e^x): hence the name;
    monitored value only (gradient uses the closed form).
  - **Numerically stable branch-wise gradient** (expm1 forms) finite for all
    float inputs — fixes the naive-autograd NaN at p≳88.
    [artifact: losses/log_quadratic_bregman.py, algorithms/spence.py,
     tests/losses/ symbolic verification]
  - Uncertainty interpretation (gamma-like) — to develop.
- **Comparison table:** MSE / Itakura–Saito / Spence × {under-predict tail,
  over-predict tail, scale-inv, uncertainty model, minimizer=V}.

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
1. Fig 1 (teaser): gradient magnitude vs prediction error for MSE / IS /
   Spence — the two-sided quadratic-tail story. (Generatable from the loss.)
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
2. §3 exp-MSE pathology + Fig 1 (gradient-vs-error, generatable from the loss).
3. §2 Setting (H-martingale targets) + App. D proofs + App. A Spence derivation.
4. §5 off-policy loss ablation (Table 1) — pull numbers from bs4_moons/dim_scaling.
5. §7 experiments (dim-scaling Fig 3, 2-moons Fig 2, GMM-40 Table 2).
6. §6 sparsity + best method; App. B zoo.
7. §1 intro + §8 related + §9 conclusion last, once the story is fixed.

# Building the PDF

No system LaTeX; use the self-contained `tectonic` engine:

    tectonic -X compile main.tex --outdir build

(one-time: download the static binary from the tectonic GitHub releases).
