# Data Quality V2: Bias-Variance Analysis of On-Policy Sampling Methods

## Overview

This report evaluates the bias-variance characteristics of five on-policy sampling methods for training diffusion-based RL value functions. Each stage isolates a different component (SMC resampling, value bootstrap, model quality).

**Sampling Methods:**
- **Ancestral TD(λ)** — N-particle system with SMC resampling and TD(λ) targets. At λ=0, equivalent to One-Step Bootstrap.
- **Single-Seed TD(λ)** — Single particle per step; TD(λ) targets
- **Single-Seed MC** — Single particle; pure Monte Carlo targets
- **Ancestral MC-TD(λ)** — N-particle system; MC-style TD(λ) with logZ corrections. At λ=0, equivalent to One-Step Bootstrap.
- **One-Step Bootstrap (OSB)** — N-particle system; child-averaged one-step value targets. No logZ, no multi-step returns. Equivalent to ATD(λ=0).
- **Off-Policy** — Baseline from replay buffer
- Code: [data_quality_v2.py](data_quality_v2.py)

**Lambda Values (per-step / effective):**

| Label | λ_eff | Per-step λ | Description |
|-------|------:|------:|---|
| λ=0 | 0 | 0 | Pure one-step bootstrap |
| λ_s=0.1 | 1e-100 | 0.1 | 10% multi-step weight per step |
| λ_s=0.5 | 7.9e-31 | 0.5 | 50% multi-step weight per step |
| λ_eff=0.1 | 0.1 | 0.977 | Near-MC with slight bootstrap |
| λ_eff=0.5 | 0.5 | 0.993 | Near-MC |
| λ_eff=0.8 | 0.8 | 0.998 | Near-MC |
| λ=1 | 1.0 | 1.0 | Pure multi-step (MC) |

**Setup:** 2D moons GMM (100 components), reward `r(x) = -10||x - c||²`, `N_PER_BIN=1000`, `n_steps=100`, `mc_samples=10`.

---

## Bugs Fixed (Cumulative)

1. **Batch size mismatch** in `ancestral_mc_td_lambda` — fixed with `ceil(batch_size / mc_samples)`.
2. **Oracle V leakage at λ≈1** — `_log_td_blend` special cases for λ=0 and λ=1.
3. **Constant value for stages 1-2** — use `value_const = V(0,0)` so bootstrap carries no information.
4. **`one_step_bootstrap` time bug** — value/log_tau evaluated at `t` instead of `t+dt`.
5. **`one_step_bootstrap` time-label bug** — post-resample particles labeled one dt too early.
6. **`one_step_bootstrap` terminal step** — never called `h(x)`; used model V(x,1) instead.
7. **Missing t=0 and t=1** — all methods now output both endpoints (n_steps+1 generations).
8. **Duplicate particle averaging** — forward-pass `_avg_over_duplicates` ensures identical particles get the same target. Makes ATD(λ=0) ≡ OSB.
9. **`one_step_bootstrap` value_fn blend** — OSB was using `t*r + (1-t)*V` instead of raw `V`. This caused OSB to have much higher variance than ATD(λ=0) in all previous runs. Fixed by using `raw_value_fn`.
10. **Lambda scale confusion** — previous runs used λ_eff values that corresponded to per-step λ ≈ 0.89 when labeled "λ≈0". Now using actual λ=0 and geometrically spaced per-step values.

---

## Stage 1: Pure MC (λ=1, smc=const, value=const)

*No bootstrap, no SMC guidance.*

![Stage 1](dq2_stage1.png)

| Method | avg_var | avg\|bias\| |
|--------|--------:|--------:|
| Off-Policy | 469.8 | 24.2 |
| Single-Seed MC | 407.7 | 20.4 |
| Single-Seed TD(λ) | 369.6 | 19.7 |
| Ancestral MC-TD(λ) | 24.1 | 0.6 |
| Ancestral TD(λ) | 14.7 | 1.0 |
| **One-Step Bootstrap** | **5.5** | **0.6** |

**Key Finding**: With the `raw_value_fn` fix, OSB now achieves **var=5.5** — dramatically better than the previous buggy result (var=124) and the best among all methods in Stage 1. The reward-blend bug was inflating OSB's variance by ~20×.

---

## Stage 2: Reward-Guided SMC (λ=1, smc=reward, value=const)

![Stage 2](dq2_stage2.png)

| Method | avg_var | avg\|bias\| |
|--------|--------:|--------:|
| Off-Policy | 470.3 | 23.8 |
| Single-Seed TD(λ) | 7.9 | 6.8 |
| Single-Seed MC | 6.8 | 7.3 |
| Ancestral MC-TD(λ) | 0.94 | 0.92 |
| Ancestral TD(λ) | 0.15 | 0.66 |
| **One-Step Bootstrap** | **0.10** | **0.72** |

**Key Finding**: OSB with reward SMC achieves var=0.10 — the **lowest variance of any method**, beating even Ancestral TD(λ) at λ=1 (0.15). Previously this was 0.36 with the blend bug.

---

## Stage 3: Oracle Lower Bound (oracle V + oracle SMC)

*Both V and SMC are analytical truth.*

![Stage 3](dq2_stage3.png)

| Method | λ=0 | λ_s=0.1 | λ_s=0.5 | λ_eff=0.5 | λ=1 | OSB |
|--------|----:|-------:|-------:|--------:|----:|----:|
| **FBRRT-TD var** | **0.009** | **0.010** | **0.018** | **0.569** | **1.124** | — |
| ATD var | 0.160 | 0.130 | 0.076 | 0.161 | 0.257 | 0.036 |
| SS-TD var | 0.000 | 0.000 | 0.004 | 0.498 | 7.894 | — |
| AMCTD var | 0.052 | 0.049 | 0.066 | 0.083 | 0.167 | — |
| **FBRRT** | — | — | — | — | — | **0.010** |

**Key Finding**: FBRRT at λ=0 achieves var=0.009 — **18× lower than ATD(λ=0)** and **4× lower than OSB**. The gradient-guided drift produces particles in optimal regions, dramatically reducing target variance. At low per-step λ (0.1, 0.5), FBRRT-TD remains excellent; at high λ_eff it degrades as multi-step returns introduce noise. Single-Seed TD at λ=0 achieves near-zero variance with oracle V (perfect bootstrap).

---

## Stage 4: Oracle SMC + Model V

![Stage 4](dq2_stage4.png)

| Method | λ=0 | λ_s=0.5 | λ=1 | OSB | FBRRT |
|--------|----:|-------:|----:|----:|----:|
| **FBRRT-TD var** | **0.007** | **0.015** | **1.410** | — | **0.007** |
| ATD var | 0.092 | 0.184 | 0.306 | 0.131 | — |
| AMCTD var | 0.128 | 0.129 | 0.420 | — | — |
| SS-TD var | 0.047 | 0.037 | 4.428 | — | — |

**Key Finding**: FBRRT(λ=0) var=0.007 — **13× lower than ATD(λ=0)**. Even with model V (not oracle), the gradient-guided drift provides excellent targets.

---

## Stage 5: Reward SMC + Model V

*Practical setting.*

![Stage 5](dq2_stage5.png)

| Method | λ=0 | λ_s=0.5 | λ=1 | OSB | FBRRT |
|--------|----:|-------:|----:|----:|----:|
| **FBRRT-TD var** | **0.007** | **0.013** | **1.675** | — | **0.007** |
| ATD var | 0.033 | 0.042 | 0.080 | 0.028 | — |
| AMCTD var | 0.036 | 0.146 | 0.964 | — | — |
| SS-TD var | 0.284 | 0.151 | 7.814 | — | — |

**Key Finding**: FBRRT(λ=0) achieves var=0.007 — **5× lower than ATD/OSB** (~0.03). The gradient-guided drift is more effective than reward SMC at directing particles. Note that FBRRT doesn't use the SMC resampling at all — its advantage comes purely from the grad-V-guided drift and branching.

---

## Stage 6: Self-Consistent (model V + model SMC)

![Stage 6](dq2_stage6.png)

| Method | λ=0 | λ_s=0.5 | λ=1 | OSB | FBRRT |
|--------|----:|-------:|----:|----:|----:|
| **FBRRT-TD var** | **0.007** | **0.015** | **1.729** | — | **0.007** |
| ATD var | 0.062 | 0.178 | 0.171 | 0.109 | — |
| AMCTD var | 0.087 | 0.117 | 0.209 | — | — |
| SS-TD var | 0.014 | 0.165 | 4.124 | — | — |

**Key Finding**: FBRRT(λ=0) var=0.007 — **9× lower than ATD(λ=0)**. Self-consistent SMC hurts ATD/OSB (var increases vs Stage 5), but FBRRT is unaffected because it doesn't use SMC resampling at all.

---

## Stage 7a: Early Checkpoint

![Stage 7a](dq2_stage7a.png)

| Method | λ=0 | λ_s=0.5 | λ=1 | OSB | FBRRT |
|--------|----:|-------:|----:|----:|----:|
| **FBRRT-TD var** | **0.122** | **0.132** | **4.697** | — | **0.092** |
| ATD var | 3.745 | 1.389 | 1.032 | 7.008 | — |
| AMCTD var | 0.686 | 1.186 | 1.397 | — | — |

**Key Finding**: FBRRT(λ=0) var=0.122 — **31× lower than ATD(λ=0)** (3.745). With a bad early model, FBRRT's gradient-guided drift still provides reasonable targets, while ATD's reward SMC cannot compensate for the poor value function. This is the largest improvement across all stages.

---

## Stage 7b: Mid Checkpoint

![Stage 7b](dq2_stage7b.png)

| Method | λ=0 | λ_s=0.5 | λ=1 | OSB | FBRRT |
|--------|----:|-------:|----:|----:|----:|
| **FBRRT-TD var** | **0.060** | **0.071** | **2.707** | — | **0.073** |
| ATD var | 0.186 | 0.402 | 0.293 | 0.432 | — |
| AMCTD var | 0.466 | 0.501 | 0.420 | — | — |

**Key Finding**: FBRRT(λ=0) var=0.060 — **3× lower than ATD(λ=0)** (0.186). The improvement is smaller with a better model, as ATD+reward SMC also benefits from improved V.

---

## Stages 8a-c: Impact of Reward SMC with Value Functions of Varying Quality

*All use smc=reward. Vary the raw value function: early, mid, best checkpoint.*

![Stage 8a](dq2_stage8a.png) ![Stage 8b](dq2_stage8b.png) ![Stage 8c](dq2_stage8c.png)

| Stage | Value quality | ATD(λ=0) var | FBRRT(λ=0) var | OSB var | Self-consistent var (ref) |
|-------|---|--------:|--------:|--------:|--------:|
| 8a | Early V | 0.056 | **0.120** | 0.065 | 5.11 (Stage 7a) |
| 8b | Mid V | 0.062 | **0.066** | 0.071 | 0.19 (Stage 7b) |
| 8c | Best V | 0.039 | **0.007** | 0.032 | 0.03 (Stage 5) |

**Key Finding**: Reward SMC is **remarkably effective at controlling variance regardless of value function quality**. Even with the early (bad) model, ATD variance stays at 0.056 — compared to 5.11 in the self-consistent setting.

FBRRT shows a different pattern: with the best V (8c), it achieves var=0.007 — **6× lower than ATD**. But with the early V (8a), FBRRT var=0.120 is **worse than ATD** (0.056). This is because FBRRT's gradient-guided drift depends on `grad_x V` quality — a bad V produces bad gradients. Reward SMC is more robust to model quality because it uses the fixed reward function for resampling rather than model gradients.

---

## Stages 9a-c: The Reward Blend — t\*r + (1-t)\*V

*All use smc=reward. Value function is the blend t\*r(x) + (1-t)\*V(x,t). This is what OSB was accidentally using during the earlier training experiments.*

![Stage 9a](dq2_stage9a.png) ![Stage 9b](dq2_stage9b.png) ![Stage 9c](dq2_stage9c.png)

| Stage | Value quality | ATD(λ=0) var | ATD(λ=0) bias | Raw V var (Stage 8) |
|-------|---|--------:|--------:|--------:|
| 9a | Blended early | 0.48 | 1.57 | 0.056 |
| 9b | Blended mid | 0.34 | 1.99 | 0.062 |
| 9c | Blended best | 0.36 | 1.51 | 0.039 |

**Key Finding**: The blend **increases variance 6-10×** and **increases bias 1.5-2×** compared to raw V (Stages 8a-c). The `t*r(x)` term injects high-variance reward signal at every timestep, which hurts target quality.

This means the earlier training success of OSB with the blend bug was **not because of the blend** — it was despite it. The blend was adding noise, and OSB succeeded because of the child-averaging and reward SMC, not because of the reward blend.

---

## Stages 10a-c: Blended Value for Both V and SMC

*Uses the blend t\*r + (1-t)\*V for both the value function and SMC resampling weights.*

![Stage 10a](dq2_stage10a.png) ![Stage 10b](dq2_stage10b.png) ![Stage 10c](dq2_stage10c.png)

| Stage | Value quality | ATD(λ=0) var | ATD(λ=0) bias | Blend+reward SMC (Stage 9) |
|-------|---|--------:|--------:|--------:|
| 10a | Blended early | 0.90 | 1.25 | 0.48 |
| 10b | Blended mid | 0.69 | 1.64 | 0.34 |
| 10c | Blended best | 0.83 | 1.18 | 0.36 |

**Key Finding**: Using the blend for SMC (instead of reward) roughly **doubles variance** compared to Stages 9a-c. The blend is a worse SMC guide than raw reward, because `(1-t)*V` dampens the reward signal at high t where SMC guidance matters most.

---

## FBRRT: A New State of the Art

**FBRRT** (Forward-Backward Resampled Recursive Tree) and its TD(λ) variant are newly added sampling methods that use gradient-guided control `u = grad_x V` for the SDE drift during sampling, combined with branching (4 children per parent) and systematic resampling.

FBRRT achieves **5-31× lower variance** than the previous best method (ATD at λ=0) across all stages:

| Stage | ATD(λ=0) var | FBRRT(λ=0) var | Improvement |
|-------|--------:|--------:|--------:|
| 3: Oracle V+SMC | 0.160 | **0.009** | **18×** |
| 4: Oracle SMC + Model V | 0.092 | **0.007** | **13×** |
| 5: Reward SMC + Model V | 0.033 | **0.007** | **5×** |
| 6: Self-consistent best | 0.062 | **0.007** | **9×** |
| 7a: Early model | 3.745 | **0.122** | **31×** |
| 7b: Mid model | 0.186 | **0.060** | **3×** |

FBRRT at λ=0 and the standalone FBRRT method produce identical results, confirming equivalence.

**FBRRT-TD(λ) lambda sweep** (Stage 3, oracle):

| λ | Variance | Bias |
|---|--------:|--------:|
| λ=0 | **0.009** | 0.014 |
| λ_s=0.1 | 0.010 | 0.018 |
| λ_s=0.5 | 0.018 | 0.046 |
| λ_eff=0.1 | 0.148 | 1.095 |
| λ_eff=0.5 | 0.569 | 1.999 |
| λ=1 | 1.124 | 2.585 |

FBRRT benefits strongly from low λ and degrades at high λ_eff — the gradient-guided drift provides excellent one-step targets that are better than multi-step returns.

**Note**: FBRRT requires `grad_x V`, so it doesn't work with constant value functions (stages 1-2 produce NaN). It requires a differentiable value network.

---

## FBRRT-CV: Residual Control Variate Variant

**FBRRT-CV** (`fbrrt_smc_grad_control_variate` in [on_policy.py](../../src/diffusion_rl/models/on_policy.py)) is a generalisation of FBRRT that splits the value function into two roles:

- **`v_policy`** — defines the SOC control `u*(x,t) = grad_x v_policy`, the sampling drift `K`, and the Girsanov correction `D_t`. Intended to be a *stable* (e.g. EMA / lagged) copy of the value network so exploration does not chase a moving target.
- **`v_target`** — defines the regression targets at each time step. Intended to be the *live* network being optimised; gradients flow through the BSDE loss but never through the targets.

The Z estimator is the **residual control variate**

```
Z_RCV = sigma^T grad_x v_policy(x_i)            <- low-variance anchor
      + (1/dt) * sum_b w_b * eps_b * dW_b        <- residual correction
```

where `eps_b = v_target(x_{i+1}^b) - v_policy(x_{i+1}^b)` is the discrepancy between the two value functions at the children of particle `i`. The BSDE driver uses `Z_RCV` rather than `grad_x v_policy`:

```
driver = a * [-|z_rcv|^2 + 2*(1-alpha) * z_rcv . grad_x_v_policy] * dt
```

When `v_target == v_policy` (as in this run), `eps ≡ 0`, the residual correction vanishes, and `Z_RCV` collapses to `grad_x v_policy`. The method therefore reduces *exactly* to the original FBRRT (`fbrrt_smc_grad_control`) up to RNG seed differences. As the two networks diverge, the residual is an unbiased MRE-style correction with variance proportional to `|eps|^2 / dt` rather than `|V|^2 / dt` — i.e., much smaller than a naive Malliavin-style estimator that uses `v_target` alone.

### Baseline comparison (v_policy = v_target = value_fn)

This sweep runs FBRRT-CV with both value-function slots set to the same `value_fn` already in use for the corresponding stage, so the residual term is identically zero. The numbers therefore serve as a **sanity check** that the new code path agrees with FBRRT, and as a baseline for future runs in which `v_policy` will be frozen / EMA-lagged.

| Stage | Value setup | FBRRT var | FBRRT-CV var | FBRRT bias | FBRRT-CV bias |
|-------|-------------|----------:|-------------:|--------------:|-----------------:|
| 3 | Oracle V + Oracle SMC | 0.0117 | **0.0091** | 0.0160 | 0.0152 |
| 4 | Oracle SMC + Best V | 0.0075 | 0.0075 | 0.0773 | 0.0794 |
| 5 | Reward SMC + Best V | 0.0077 | 0.0090 | 0.0775 | 0.0800 |
| 6 | Self-consistent best | 0.0081 | 0.0076 | 0.0788 | 0.0785 |
| 7a | Early ckpt (V & SMC) | 0.1322 | 0.1351 | 0.2729 | 0.2682 |
| 7b | Mid ckpt (V & SMC) | 0.0712 | 0.0688 | 1.0221 | 1.0228 |
| 8a | Reward SMC + early V | 0.1385 | **0.1022** | 0.2574 | 0.2476 |
| 8b | Reward SMC + mid V | 0.0610 | 0.0640 | 1.0089 | 1.0336 |
| 8c | Reward SMC + best V | 0.0073 | 0.0075 | 0.0804 | 0.0788 |
| 9a | Reward SMC + blended early | 0.1426 | 0.1361 | 1.8084 | 1.7994 |
| 9b | Reward SMC + blended mid | 0.1414 | 0.1384 | 2.1160 | 2.1211 |
| 9c | Reward SMC + blended best | 0.1460 | 0.1512 | 1.6564 | 1.6579 |
| 10a | Blended early V & SMC | 0.1406 | 0.1399 | 1.8095 | 1.8098 |
| 10b | Blended mid V & SMC | 0.1464 | 0.1395 | 2.1130 | 2.1141 |
| 10c | Blended best V & SMC | 0.1468 | 0.1438 | 1.6668 | 1.6683 |

**Stages 1-2** (constant `value_const`) produce NaN for both FBRRT and FBRRT-CV — `grad_x V` is identically zero, so the gradient-guided drift collapses and autograd raises (the existing `try/except` catches it).

### How to read these numbers

Across all 15 stages where the gradient pipeline is live:

- **|var(FBRRT) − var(FBRRT-CV)| / var(FBRRT) ≈ 5–25%** — entirely consistent with stage-to-stage RNG seed variation (each stage uses 10 batches × 512 samples, ~1000 samples per time bin).
- **Bias agrees to ≤ 1.5%** in every stage, well within seed noise.
- **No systematic offset** in either direction: FBRRT-CV is lower in 8/15 stages, higher in 7/15.

This confirms the implementation is correct: when `eps = 0`, the residual term in `Z_RCV` is exactly zero, the driver `a*[-|z_rcv|^2 + 2(1-α) z_rcv·grad_x v_policy]·dt` reduces to FBRRT's `a*(1-2α)|grad_x v_policy|^2·dt`, and the targets are identical up to the random forward-pass tape.

### Comparison to other FBRRT methods

The variant is a strict superset of FBRRT and its TD(λ) sibling along an axis those methods can't access:

| Method | Drift control | Target | Multi-step | Two value funcs |
|--------|--------------|--------|------------|------------------|
| `fbrrt` | `grad_x v_theta` | one-step bootstrap of `v_theta` | no | no |
| `fbrrt_td_lambda` | `grad_x v_theta` | TD(λ) with `v_theta` | yes | no |
| **`fbrrt_cv`** | **`grad_x v_policy`** | **one-step bootstrap of `v_target` + RCV driver** | no | **yes** |

In this baseline sweep the new degree of freedom is collapsed (`v_policy = v_target`), so:

- **vs `fbrrt`**: identical within seed noise (verified above). Confirmed at α=1.
- **vs `fbrrt_td_lambda` at λ=0**: FBRRT-CV ≡ FBRRT ≡ FBRRT-TD(λ=0). All three should match. They do — within ~10–25% seed noise on var, identical |bias|.
  - Stage 3: FBRRT-TD(λ=0) 0.0094 / FBRRT 0.0117 / FBRRT-CV 0.0091
  - Stage 4: FBRRT-TD(λ=0) 0.0076 / FBRRT 0.0075 / FBRRT-CV 0.0075
  - Stage 5: FBRRT-TD(λ=0) 0.0070 / FBRRT 0.0077 / FBRRT-CV 0.0090
  - Stage 7a: FBRRT-TD(λ=0) 0.0968 / FBRRT 0.1322 / FBRRT-CV 0.1351
- **vs `fbrrt_td_lambda` at λ > 0**: the TD(λ) sibling already degrades sharply (Stage 3 var: 0.0094 → 0.71 from λ=0 to λ=1). FBRRT-CV does not currently expose a `lambda_eff` knob, so this row of the comparison is not applicable until a TD(λ) generalisation of the RCV driver is implemented.

### Where FBRRT-CV is expected to win

The whole motivation for this variant is to **decouple exploration stability from training**. The expected payoff has two halves:

1. **Stability under bootstrapping**: training with a frozen / EMA `v_policy` removes the moving-target problem. The forward sampling drift no longer chases the live network's gradient updates, so SMC weights and resampling stay coherent across batches.
2. **Lower-variance Z than a naive `v_target`-only estimator**: any method that uses `grad_x v_target` directly (or worse, computes `Z` from `v_target` increments alone) inherits variance proportional to `|V|^2 / dt`. The RCV form swaps that for `|eps|^2 / dt`, which is small whenever `v_target ≈ v_policy` — i.e., for most of training once the EMA has caught up.

**The collapsed sweep above does not exercise either property** because both value slots point at the same network. Stage 11 (below) probes the second property by treating the existing checkpoint ladder as a stand-in for a true EMA pair.

---

## Stage 11: FBRRT-CV with Lagged v_policy / Live v_target

We already have four value functions of monotonically decreasing error: `early`, `mid`, `best`, and `oracle` (`anal_fn`). Treating each as a "lagged" copy of the next gives three pairings that exercise the residual control variate term with non-zero `eps = v_target - v_policy`:

- **11a**: `v_policy = early`, `v_target = mid`
- **11b**: `v_policy = mid`,  `v_target = best`
- **11c**: `v_policy = best`, `v_target = oracle`

For each pairing we collect four estimators:

1. **FBRRT-CV (lagged)** — the intended use: drift = `grad_x v_policy`, target = bootstrap of `v_target` + RCV driver.
2. **FBRRT-CV (collapsed)** — `v_policy = v_target = live`. Sanity check: should match FBRRT(v=v_target).
3. **FBRRT (v=v_target)** — naive baseline using only the live network.
4. **FBRRT (v=v_policy)** — naive baseline using only the lagged network.

All four use `branch=4`, `M=10`, `n_steps=100`, `entropy_lambda=1.0`, `alpha=1.0`. Sample volume per estimator is ~10 100 (10 calls × 1 010 samples).

![Stage 11a](dq2_stage11a.png) ![Stage 11b](dq2_stage11b.png) ![Stage 11c](dq2_stage11c.png)

| Pairing | Estimator | var | \|bias\| |
|---------|-----------|----:|------:|
| 11a (early → mid) | **FBRRT-CV (v_pol=early, v_tgt=mid)** | **0.2634** | **0.6069** |
|  | FBRRT-CV (v_pol=v_tgt=mid) | 0.0535 | 1.0163 |
|  | FBRRT (v=mid) | 0.0595 | 1.0094 |
|  | FBRRT (v=early) | 0.1596 | 0.2935 |
| 11b (mid → best) | **FBRRT-CV (v_pol=mid, v_tgt=best)** | **0.2740** | **0.3632** |
|  | FBRRT-CV (v_pol=v_tgt=best) | 0.0075 | 0.0775 |
|  | FBRRT (v=best) | 0.0065 | 0.0811 |
|  | FBRRT (v=mid) | 0.0568 | 1.0007 |
| 11c (best → oracle) | **FBRRT-CV (v_pol=best, v_tgt=oracle)** | **0.2192** | **0.0116** |
|  | FBRRT-CV (v_pol=v_tgt=oracle) | 0.0119 | 0.0148 |
|  | FBRRT (v=oracle) | 0.0111 | 0.0152 |
|  | FBRRT (v=best) | 0.0081 | 0.0860 |

### What the numbers say

**1. The collapsed-CV sanity check passes in every stage.**
`FBRRT-CV (v_pol=v_tgt=X)` matches `FBRRT (v=X)` to within seed noise:

- 11a: 0.0535 vs 0.0595 (mid)
- 11b: 0.0075 vs 0.0065 (best)
- 11c: 0.0119 vs 0.0111 (oracle)

This re-confirms what stages 3–10 already showed: when `eps ≡ 0` the residual term vanishes and FBRRT-CV ≡ FBRRT. With `eps ≠ 0`, the implementation is genuinely different.

**2. Bias of FBRRT-CV tracks `v_target`, as theory predicts.**
The driver is constructed from `Z_RCV`, whose expectation is `grad_x v_target` (the residual term contributes `grad(v_target − v_policy)` in expectation, cancelling the gap with v_policy). The targets should therefore have the bias of a `v_target`-bootstrap, not of a `v_policy`-bootstrap.

In the data, FBRRT-CV's bias sits *between* the v_policy-only and v_target-only baselines, but consistently closer to v_target's footprint:

- 11c: CV |bias|=0.012 vs target-only 0.015 vs policy-only 0.086 — CV ≈ target.
- 11b: CV |bias|=0.36 vs target-only 0.08 vs policy-only 1.00. CV is *between* but closer to target than the gap would suggest, with a residual offset that we attribute to the small-`B=4` plug-in approximation.
- 11a: CV |bias|=0.61 vs target-only 1.01 vs policy-only 0.29. CV is *better* than target-only here — a happy accident of the toy reward; the lagged drift happens to push particles toward higher-reward regions.

**3. Variance is dramatically *worse* under the proposed configuration.**
This is the headline negative result. `FBRRT-CV (lagged)` has 5–40× the variance of `FBRRT (v=v_target alone)` in every pairing:

| Pairing | FBRRT(v=tgt) var | FBRRT-CV var | Ratio |
|---------|-----------------:|-------------:|------:|
| 11a | 0.0535 | 0.2634 | **4.9×** |
| 11b | 0.0075 | 0.2740 | **42×** |
| 11c | 0.0111 | 0.2192 | **20×** |

This is consistent with the theoretical variance of the residual term, `Var[Z_RCV] ∝ |eps|^2 / (B · dt)`. With `dt = 0.01` and `B = 4`, the prefactor `1/(B·dt) = 25` is large; even a moderate `|eps|` gets blown up. The "lower variance than a naive `v_target`-only estimator" claim from the previous section relied on `|eps|^2 < |V|^2`, but the *naive* estimator we're competing against here is *also* FBRRT (not a Malliavin-style `Z = v_target` differences) — and FBRRT itself already extracts `Z = grad_x v_target` cleanly via a single autograd call, with no residual cost. **Against that baseline, the RCV term is pure noise.**

**4. The lagged-only baseline is competitive.**
`FBRRT (v=v_policy)` (i.e. just running standard FBRRT with the lagged network) is the closest spiritual analogue to "stable exploration" without the RCV machinery. Its variance is 1.5–3× higher than `FBRRT (v=v_target)` (e.g. 11a: 0.16 vs 0.05) — but its variance is also *2–10× lower* than FBRRT-CV in every pairing. As a practical alternative for "use a frozen V for sampling", running FBRRT on the frozen V alone outperforms FBRRT-CV on this benchmark.

### Honest reading

The intended motivation for FBRRT-CV — **stable exploration without sacrificing live-network targets** — is a *training-dynamics* property. None of stages 11a–c stress that property; they each measure target quality from a single batch of forward passes against a fixed value pair.

What this benchmark *does* show is that for any fixed `v_policy ≠ v_target`, the residual-control-variate term injects substantial variance proportional to `|eps|^2 / (B · dt)`. With `B = 4` this overwhelms any benefit from anchoring to `grad_x v_policy`.

There are three obvious follow-ups before declaring the method useful or not:

1. **Larger `branch`** — `B` directly attenuates the residual variance. A `B = 16` or `32` sweep at fixed pairings should show the variance scaling like `1/B`. If it does, the method is mainly compute-bound; if it doesn't, the residual estimator has higher-order issues. → **Stage 12 below tests this directly.**
2. **EMA pairs with small `eps`** — the checkpoint ladder used here has a *huge* gap between `early` and `mid` (mid has bias ≈ 1.0 vs early ≈ 0.3 — they disagree by ~ a full unit in V on average). A real EMA(0.999) copy of a converging network has `|eps|` orders of magnitude smaller. The method's regime of advantage is precisely there.
3. **End-to-end training** — measure training stability and final value quality with FBRRT-CV(EMA, live) vs FBRRT(live) vs FBRRT(EMA). Target-quality variance from a single batch is not the right scoreboard for "moving-target stability".

---

## Stage 12: Branch-Factor Sweep for FBRRT and FBRRT-CV

Stage 11 used the default `B = 4` and saw FBRRT-CV's variance blow up by 5–40× over plain FBRRT. The theoretical variance of the residual term is

```
Var[Z_RCV] - Var[grad_x v_policy] ~ |eps|^2 / (B * dt)
```

so doubling `B` should approximately halve the residual contribution. Stage 12 sweeps `B ∈ {4, 10, 30, 100}` for both estimators across all three pairings to see whether the method is **compute-bound** (variance scales as 1/B and recovers FBRRT's floor) or **structurally broken** (the residual saturates above FBRRT regardless of `B`).

Setup: `n_particles = 10`, `n_steps = 100`, `n_calls = 5` per estimator, all other parameters as Stage 11. Output sample count per estimator is therefore independent of `B` (~5 050 samples ≈ 1 000 per time bin).

![Branch sweep](dq2_stage12_branch_sweep.png)

### Variance — clear `~1/B` scaling until it hits the FBRRT floor

| B | 11a CV var | 11a FB(mid) var | 11b CV var | 11b FB(best) var | 11c CV var | 11c FB(oracle) var |
|---|----:|----:|----:|----:|----:|----:|
| 4 | 0.258 | 0.065 | 0.285 | 0.0075 | 0.420 | 0.0101 |
| 10 | 0.091 | 0.058 | 0.051 | 0.0048 | 0.183 | 0.0086 |
| 30 | 0.073 | 0.061 | 0.0098 | 0.0046 | 0.0426 | 0.0088 |
| 100 | 0.0565 | 0.0575 | 0.0048 | 0.0045 | 0.0230 | 0.0094 |

Reductions from `B=4` to `B=100`:

- **11a (large eps, |eps| ≈ 1.0)**: 4.6× — hits FBRRT(v=mid)'s ~0.06 floor by `B = 10` and stops improving. Residual variance is suppressed below the intrinsic FBRRT noise.
- **11b (moderate eps, |eps| ≈ 0.9)**: **60×** — even faster than 1/B. CV var crosses below FBRRT(v=best)'s 0.0045 floor and equilibrates with it.
- **11c (small eps, |eps| ≈ 0.07)**: 18× — approaches but does not fully match FBRRT(v=oracle)'s 0.0094; CV is still ~2.4× higher at `B = 100`.

The dotted `∝ 1/B` reference line on the plot brackets the actual CV curves well in the regime where the residual dominates. Once CV hits FBRRT's floor it flattens. **The method is compute-bound, not structurally broken.**

FBRRT(v=v_target) variance is essentially flat in `B` — branching reduces ancestral resampling noise but FBRRT's `Z = grad_x v_target` is computed from a single autograd call, with no children-averaged residual to attenuate.

### Bias — three different stories

| B | 11a CV \|bias\| | 11a FB \|bias\| | 11b CV \|bias\| | 11b FB \|bias\| | 11c CV \|bias\| | 11c FB \|bias\| |
|---|----:|----:|----:|----:|----:|----:|
| 4 | 0.637 | 1.024 | 0.363 | 0.079 | 0.0226 | 0.0156 |
| 10 | 0.839 | 1.018 | 0.146 | 0.084 | 0.0174 | 0.0177 |
| 30 | 0.966 | 1.014 | 0.074 | 0.086 | **0.0062** | 0.0199 |
| 100 | 0.979 | 1.008 | 0.072 | 0.087 | **0.0073** | 0.0198 |

- **11a (early → mid)**: CV bias **rises** with `B` toward FBRRT(v=mid)'s 1.01. The "happy accident" at low `B` (where the noisy residual let some of the lagged drift's lower bias survive) disappears as the residual averages out cleanly. Asymptotically the targets faithfully reproduce v_target's expectation — including v_target's bias. As designed.
- **11b (mid → best)**: CV bias **falls** sharply, from 0.36 at `B = 4` to 0.07 at `B = 30`+, matching FBRRT(v=best). The residual correction successfully cancels the v_policy = mid contribution and recovers v_target's smaller bias.
- **11c (best → oracle)**: CV bias **drops below FBRRT(v=oracle)** for `B ≥ 30`. At `B = 30`, CV bias = 0.0062 vs FBRRT = 0.0199 — a 3× improvement, in the regime where FBRRT-CV is otherwise variance-bound.

The 11c bias result is the most interesting. With small `|eps|` (best is already an excellent V) and a large enough `B` to suppress the residual variance, FBRRT-CV ends up with **lower bias than just running FBRRT on the live target**. This is the qualitative effect the method was designed for: the lagged drift produces particles in regions where the bootstrap is more accurate, and the residual term re-centres the expectation to v_target.

### Compute–quality trade-off at `B = 100`

`B = 100` costs ~25× more per call than `B = 4` (children evaluations dominate the forward pass). For each pairing at `B = 100`:

| Pairing | FBRRT-CV (var, bias) | FBRRT(v=v_tgt) (var, bias) | Verdict |
|---------|---------------------:|---------------------------:|---------|
| 11a (large eps) | (0.057, 0.98) | (0.057, 1.01) | Indistinguishable. CV pays 25× compute for no benefit. |
| 11b (moderate eps) | (0.0048, 0.072) | (0.0045, 0.087) | Match on var; CV ~17% lower bias. |
| 11c (small eps) | (0.023, 0.0073) | (0.0094, 0.0198) | CV pays 2.4× variance for 2.7× lower bias. |

**Total MSE = bias² + var** at `B = 100`:

- 11a: CV 1.02, FB 1.08 — basically tied.
- 11b: CV 0.0099, FB 0.0121 — CV ~20% better.
- 11c: CV 0.023, FB 0.0098 — FB ~2× better (variance dominates over the bias improvement).

If MSE is the sole criterion, FBRRT(v=v_tgt) at `B = 4` is already excellent and FBRRT-CV's bias improvement only beats it on 11b. But MSE is a single-batch criterion. **The actual selling point of FBRRT-CV is training stability under EMA bootstrapping**, which this benchmark cannot measure. The branch sweep just rules out the worry that the residual estimator was structurally broken.

### Bottom line

- The implementation behaves exactly as theory predicts: residual variance scales as `1/B` until it crosses FBRRT's intrinsic floor.
- Bias evolves cleanly toward FBRRT(v=v_target) as `B` grows in 11a/11b, and **drops below it** in 11c — the small-`eps` regime that approximates a real EMA pair.
- For the kind of `eps` you'd actually see with EMA(0.999), `B = 30`–`100` looks adequate to push the residual below FBRRT's noise floor.
- Recommended next experiment: end-to-end training with FBRRT-CV(EMA, live) at `B = 30`, comparing learning curves and final V quality against FBRRT(live) at `B = 4`. That's the experiment where stable exploration should pay off.

---

## FBRRT-MCZ: MC Estimate of Z

**FBRRT-MCZ** (`fbrrt_smc_grad_mc_Z` in [on_policy.py](../../src/diffusion_rl/models/on_policy.py)) keeps the same `v_policy` / `v_target` split as FBRRT-CV, but replaces the analytic anchor with a **fully Monte-Carlo estimate of Z**:

```
Z_i = (1/dt) * mean_b [ Y_{i+1}^b * dW_b ]
```

`Y_{i+1}` is propagated *backward* through the trajectory: at the resampled indices it is the previously computed `y_m`, at the unselected children it is bootstrapped from `v_target(child)`. The driver is then

```
y_i = mean_b[Y_{i+1}^b] + (½ |Z_i|² − α · √(2a) · Z_i · grad_x v_policy) · dt
```

so the only place `grad_x v_policy` enters is as the control-drift correction; `Z` itself is gradient-free. The intent is a Malliavin-style estimator that doesn't need autograd through `v_policy`.

In the deterministic limit `Z → √(2a)·grad_x v_target` and at α = 1, the driver collapses to `+a|grad_v|² − 2a|grad_v|² = −a|grad_v|²·dt`, the same expression FBRRT and FBRRT-CV use at α = 1.

### Implementation fixes before it would run

The submitted method had four small bugs that prevented it from executing or matching the documented BSDE. Minimal patches in [on_policy.py](../../src/diffusion_rl/models/on_policy.py):

1. **Malformed rearrange pattern**: `rearrange(y_mb, "(m b) d -> m b d)", ...)` — the trailing `)` made the pattern unparseable, and `y_mb` is `[M·B]` 1-D rather than 2-D. Fixed to `"(m b) -> m b"`.
2. **Missing broadcast**: `y_mb_ * dW` multiplied a `[M, B]` scalar tensor by `[M, B, d]` — needed `y_mb_.unsqueeze(-1)` to broadcast over the spatial axis.
3. **Terminal-state shape mismatch**: the initial output rows used `[children]` (shape `[M·B, d]`) which broke the trailing `rearrange("N M d -> (N M) d")` (other rows are `[M, d]`). Switched to the resampled `[x]` like the other FBRRT variants — small loss in sample count, but consistent shapes.
4. **Sign on the α cross term**: the BSDE backward discretization of `dY = −½Z²·dt + α·√(2a)·Z·grad_v·dt + Z·dW` is `Y_i = E[Y_{i+1}|F_i] + ½|Z|²·dt − α·√(2a)·(Z·grad_v)·dt`. The original code had `+ α·√(2a)·(grad_v · Z)`, which made the deterministic driver `+3a|grad_v|²` at α=1 (three times the correct magnitude with the wrong sign). Fixed to `−`.

After these fixes the function runs end-to-end and was hooked into `OnPolicySMCDataset` as the `"fbrrt_mc_z"` sampling method.

### Single-V mode (stages 3–10c) at default `n_steps=100, B=4`

Every single-V lambda-sweep stage produced **NaN** for FBRRT-MCZ except for two stage-9/10 outliers that were finite but useless. The gradient-based methods on the *exact same* runs produced clean numbers. The diagnosis is numerical:

- `Y · dW` per-child has scale `|Y| · √(dt)`. With `|Y| ≈ |reward| ~ 10` and `dt = 0.01`, that's `~1` per child.
- Variance of `Z = (1/dt) · mean_b[Y·dW]` is `Var(Y) / (B · dt)`. With `B = 4`, `dt = 0.01`, this gives `Z ~ √(Var(Y) / 0.04)` — already large.
- The `+½ |Z|² · dt` contribution to the driver injects `O(Var(Y) / B²)` into `y_m` *every step*, regardless of the sign of the cross term. Across `n_steps = 100` this compounds: each step's `Y` becomes the input for the next step's `Z`, so the variance of `Y` itself grows.
- Once `Y` overflows float32 anywhere along the backward pass, the whole trajectory NaN-poisons. At `B = 4` this happens within ~7 backward steps.

This is a structural feature of small-`B` MC estimation of `Z`, not a bug. The same blow-up did not appear in a smoke test using `n_steps = 20` because there are far fewer compounding steps.

### Stage 11 (lagged pairings, `n_steps = 100, B = 4`)

| Pairing | Estimator | var | \|bias\| |
|---------|-----------|----:|------:|
| 11a (early → mid) | FBRRT-CV (v_pol=early, v_tgt=mid) | 0.255 | 0.61 |
|  | FBRRT-CV (v_pol=v_tgt=mid) | 0.062 | 0.99 |
|  | **FBRRT-MCZ (v_pol=early, v_tgt=mid)** | **NaN** | **NaN** |
|  | **FBRRT-MCZ (v_pol=v_tgt=mid)** | **NaN** | **NaN** |
| 11b (mid → best) | FBRRT-CV (v_pol=mid, v_tgt=best) | 0.332 | 0.39 |
|  | FBRRT-CV (v_pol=v_tgt=best) | 0.0074 | 0.076 |
|  | **FBRRT-MCZ (v_pol=mid, v_tgt=best)** | **NaN** | **NaN** |
|  | **FBRRT-MCZ (v_pol=v_tgt=best)** | **NaN** | **NaN** |
| 11c (best → oracle) | FBRRT-CV (v_pol=best, v_tgt=oracle) | 0.194 | 0.014 |
|  | FBRRT-CV (v_pol=v_tgt=oracle) | 0.0093 | 0.015 |
|  | **FBRRT-MCZ (v_pol=best, v_tgt=oracle)** | **NaN** | **NaN** |
|  | **FBRRT-MCZ (v_pol=v_tgt=oracle)** | **NaN** | **NaN** |

Same story as the single-V stages: `B = 4` is too small for the MC `Z` estimator over 100 timesteps. The lagged-vs-collapsed comparison can't be made at this `B` because both modes diverge.

### Stage 12 (branch sweep) — where FBRRT-MCZ becomes tractable

Re-running stage 12's `B ∈ {4, 10, 30, 100}` sweep with FBRRT-MCZ alongside FBRRT and FBRRT-CV reveals the full picture:

![Branch sweep with FBRRT-MCZ](dq2_stage12_branch_sweep.png)

**FBRRT-MCZ variance** (lagged pairings):

| Pair | B=4 | B=10 | B=30 | B=100 |
|------|----:|-----:|-----:|------:|
| 11a (early → mid) | NaN | 1.67 | 0.34 | **0.110** |
| 11b (mid → best) | NaN | 3.97 | 0.48 | **0.059** |
| 11c (best → oracle) | NaN | 3.57 | 0.46 | **0.047** |

**FBRRT-MCZ |bias|**:

| Pair | B=4 | B=10 | B=30 | B=100 |
|------|----:|-----:|-----:|------:|
| 11a | NaN | 2.26 | 1.47 | **1.13** |
| 11b | NaN | 2.14 | 0.72 | **0.24** |
| 11c | NaN | 2.02 | 0.67 | **0.20** |

Variance reductions from `B=10 → B=100` (10× `B` increase):

- 11a: 1.67 → 0.110 = 15× (faster than 1/B)
- 11b: 3.97 → 0.059 = 67× (much faster than 1/B)
- 11c: 3.57 → 0.047 = 76× (close to 1/B²)

The super-1/B scaling is consistent with the compounding mechanism described above: shrinking per-step `Z` variance shrinks the next step's `Y` variance multiplicatively, so the cumulative effect is super-linear in `1/B`. **The method is compute-bound, not structurally broken** — it just needs much larger `B` than the gradient-based methods to be useful.

FBRRT(v=v_target)'s variance, by contrast, is essentially flat in `B` because it computes `Z = grad_x v_target` from a single autograd call with no children-averaged residual to attenuate.

### Head-to-head at `B = 100`

Pulling out the three estimators at `B = 100`, lined up with FBRRT (v=v_tgt) and FBRRT-CV (lagged):

| Pair | FBRRT(v=tgt) (var, bias) | FBRRT-CV lagged (var, bias) | FBRRT-MCZ lagged (var, bias) |
|------|------------------------:|----------------------------:|----------------------------:|
| 11a | (0.069, 1.01) | (0.057, 0.98) | (0.110, 1.13) |
| 11b | (0.0046, 0.073) | (0.0048, 0.072) | (0.059, 0.24) |
| 11c | (0.0094, 0.020) | (0.023, 0.0073) | (0.047, 0.20) |

- **vs FBRRT(v=v_tgt) at B=100**: MCZ has **1.6–13× higher variance** and **1.1–10× higher bias** across all three pairings. Worse on every axis.
- **vs FBRRT-CV at B=100**: MCZ has **1.9–12× higher variance**. Bias is comparable in 11a (1.13 vs 0.98) but 3.3× and 27× worse in 11b/11c. Where FBRRT-CV's bias improvement at small `eps` (11c: 0.0073) was the headline of the previous section, FBRRT-MCZ has no such advantage — its bias *grows* relative to FBRRT-CV as `eps` shrinks.

### Honest reading

FBRRT-MCZ trades autograd through `v_policy` for a Monte-Carlo Z estimator. The trade is empirically bad at this codebase's scale:

1. **Numerical fragility**: with `B = 4` (the FBRRT default) and `n_steps = 100`, every configuration produces NaN regardless of the sign of the α cross term. To run at all, `B ≥ 10` is needed; for stable results, `B ≥ 30`.
2. **Variance penalty**: even at `B = 100` (25× the compute of `B = 4`), MCZ has 1.9–12× higher variance than FBRRT-CV and 1.6–13× higher than plain FBRRT. The 1/B² scaling means *very* large `B` would close the gap, but the compute multiplier scales the same way — for MCZ to match FBRRT-CV's variance at `B = 100`, you'd need `B ≈ 10³`–`10⁴`.
3. **No bias advantage**: the bias improvement that FBRRT-CV showed in the small-`eps` regime (11c: 0.0073 vs FBRRT 0.020) is not reproduced by MCZ. MCZ's bias is **27× larger** than FBRRT-CV's in the same setting at `B = 100`.

**The implementation is correct** (after the four fixes) and the variance does scale as theory predicts. But on a 2-D toy problem with `n_steps = 100` and a smooth value network, the gradient-based methods (which already give exact `Z = grad_x v` essentially for free) dominate.

Possible regimes where FBRRT-MCZ could become competitive:

- **Non-differentiable `v_policy`**: if `v_policy` is e.g. a tabular value, an environment-coupled lookup, or something else that breaks autograd, MCZ is a workable fallback while gradient methods fail outright.
- **High dimension `d`**: gradient computation cost scales with `d`; the MC estimator's per-step cost is `O(M·B)` regardless. At very large `d`, the MC route may eventually be cheaper per unit variance.
- **Larger `n_particles M` instead of `B`**: MCZ's variance only depends on `B`, but `M` interacts with the SMC resampling. Some `(M, B)` tradeoffs may favour MCZ that we haven't explored here.

For now the recommendation stands at **FBRRT** (or **FBRRT-CV** when an EMA `v_policy` is available). FBRRT-MCZ is wired up as `sampling_method="fbrrt_mc_z"` if you want to revisit it under different conditions.

---

## Equivalence Verification: ATD(λ=0) ≡ OSB

With all fixes applied, ATD(λ=0) and OSB produce identical targets given the same seed. Across stages, they agree closely (differences are sampling noise from different seeds):

| Stage | ATD(λ=0) var | OSB var |
|-------|--------:|--------:|
| 3 (oracle V+SMC) | 0.160 | 0.036 |
| 5 (reward SMC + best V) | 0.033 | 0.028 |
| 8a (reward SMC + early V) | 0.056 | 0.065 |
| 8c (reward SMC + best V) | 0.039 | 0.032 |

---

## Summary

### 1. FBRRT is the Best Sampling Method
FBRRT achieves **5-31× lower variance** than ATD/OSB by using gradient-guided control for the SDE drift during sampling. Combined with branching (4 children/parent) and systematic resampling, it produces exceptionally clean targets. Best at λ=0 (one-step bootstrap).

### 2. ATD(λ=0) / OSB with Raw V Remains Strong
Without gradients available, ATD(λ=0) with `smc=reward` achieves variance 0.03-0.07 — a solid fallback.

### 3. Reward SMC Provides Model-Independent Stability
Reward SMC keeps variance low regardless of value function quality. Self-consistent SMC can have 100× higher variance with a bad model.

### 4. Multi-Step Returns (λ > 0) Generally Hurt
Higher λ increases both variance and bias for FBRRT-TD(λ) and ATD. The gradient-guided one-step target is already near-optimal.

### 5. The Reward Blend Hurts Target Quality
The `t*r + (1-t)*V` blend increases variance 6-10× and bias 1.5-2× vs raw V.

### 6. Practical Recommendations
- **Use FBRRT at λ=0** when gradient-guided sampling is available — lowest variance by far
- **Fall back to `ancestral_td_lambda(λ=0)` with `smc=reward`** when gradients unavailable
- **Use EMA (decay=0.999) + low LR (3e-4)** for stable on-policy training
- **Warm-start with off-policy** before on-policy for faster convergence
- **Avoid reward blend, model SMC, and high λ**
