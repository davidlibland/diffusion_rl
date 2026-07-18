# Probe 4 — TRI-TSMC trust-region mechanisms: report

**Question.** Do the trust-region mechanisms of TRI-TSMC (Wang et al. 2026,
arXiv:2605.25123) — KL-budgeted twist movement, escort-ramped concentration,
dual-tempered importance weights — improve our continuous value-training loop
at d=512? Machinery verified against analytic cases (Gaussian KL to 3
decimals, exact bisection, monotone ESS restoration); all runs error-free.

**Answer in one line: the mechanisms work as designed, and their diagnostics
mostly show our existing setup was already inside the trust region** — with
one directional win (tempered unweighting beats the clamp) and one sharpened
negative (ramped concentration doesn't rescue the oracle twist).

Setup: ssmc, d=512, 15k steps, paired seeds 0–9; ε scan {0.3, 1.0, 3.0} on
seeds 0–2, winners extended to 10 seeds. Controls: law-v2 grid (35.3±2.3),
off-policy (30.7±1.6 on these seeds), probe-2 oracle twist (16.3±1.4),
probe-3 expand_oracle (21.9±2.7).

## Results (best ε per arm, n=10)

| arm | ε | mean | paired vs its control | vs law-v2 | diagnostics |
|---|---|---|---|---|---|
| tr_twist | 0.3 | 30.9±2.8 | −4.0±2.6 vs law-v2 (p=.16) | — | **β=1.000 at all 50 updates** |
| tr_oracle_ramp | 3.0 | 20.5±1.6 | **+4.1±1.2 vs oracle_twist (p=.008)** | −14.5±3.3 (p=.002) | β saturates by epoch 1–4 at every ε |
| tr_unweight | 0.3 | 25.2±3.4 | +3.2±2.0 vs expand_oracle (p=.15, 7/10) | −9.8±2.3 | τ≈0.21 (dual strongly active) |

Scan detail: tr_twist scores were bit-identical across ε (26.7/26.7/26.7 —
see below); tr_oracle_ramp 20.1/19.2/20.9 (no ramp-speed effect);
tr_unweight 24.4/20.3/19.4 (monotone: tighter budget better).

## Readings

1. **The trust region never binds for the learned twist.** tr_twist's β hit
   1.0 at every one of ~50 updates even at ε=0.3 — the k·V twist (k≈0.13)
   moves *well under* 0.3 nats per 304-step epoch, so all ε arms ran
   bit-identically. This is a genuinely useful null: **ssmc's learned twist
   already moves far slower than any reasonable KL budget**, which
   retroactively explains why EMA/staleness knobs barely mattered for ssmc
   (they matter for ssmc-td, whose bootstrapped targets — not its twist —
   are the fast-moving object). The arm's −4.0 vs law-v2 is attributable to
   its forced config changes (live-net twist, cadence 1216 vs 4594), not to
   the trust-region logic, which was inert.
2. **Ramped concentration does not rescue the oracle twist.** ε controls the
   ramp exactly as designed (ε=0.3: β = 0.25 → 0.49 → 0.83 → 1.0 over four
   epochs; ε=3.0: instant), but every feasible ramp saturates within ~8% of
   training and all land at 19–21% — far below the 25.9–35.3% of
   unconcentrated configs. The +4.1 over probe-2's oracle twist is
   explained by the law-v2 base config (expansion active), not the ramp:
   ramp speed had no effect within the scan. The coverage-vs-concentration
   conclusion survives in graded form; a *much* slower ramp (ε ≈ 0.01–0.03,
   spanning most of training) is the remaining untested regime —
   supplementary cell running.
3. **Dual-tempered unweighting directionally beats the clamp** (+3.2, 7/10
   seeds) and the scan trend is monotone toward *tighter* budgets — raising
   the possibility that the optimum is ε → 0, i.e. **uniform weights: no
   unweighting at all** (bias from corridor sources costs less than the
   variance of correcting for them). Supplementary cell (ε=1e-4 ≈ uniform)
   running to close this.

**Methodological caveat:** our per-regen KL is estimated by reweighting the
epoch's rows — a state-marginal surrogate for the true sequential path KL,
which it underestimates. Budgets here are therefore "at least this
permissive"; the tr_twist null (β=1 always) is robust to this (the true KL
being larger would only mean the budget binds *more*, and it never came
close), but the ramp speeds are upper bounds on effective ramp slowness.

## Supplements (complete)

- **Slow ramp (ε=0.03): 20.2±1.4** — β spends most of training ramping
  (mean-over-updates 0.887) yet the outcome is statistically identical to
  the instant ramp (−0.3±0.9 vs ε=3.0, p=.75) and still −14.8 below law-v2.
  Across two orders of magnitude of ramp speed, **concentration harm is
  ramp-speed-invariant** in every reachable regime. Reading 2 closes: at
  d=512 there is no rate at which oracle-strength concentration becomes
  affordable.
- **Uniform-weight limit (ε=1e-4): 18.0±1.7** — the scan's monotone trend
  does NOT continue to ε→0. The full ordering within the oracle-source
  expansion family is a clean bias-variance U:

  | weighting | mean |
  |---|---|
  | uniform (no unweighting; biased targets) | 18.0±1.7 |
  | raw, clamped at 100× (probe 3; unbiased, high variance) | 21.9±2.7 |
  | **dual-tempered, ε=0.3 (interior optimum)** | **25.2±3.4** |

  Unweighting is earning its keep, but only in tempered form: the
  correction's bias reduction is real (uniform is worst) and its raw
  variance is real (clamp beats uniform but loses to temper by 3.3). This
  is the one adoptable mechanism from probe 4.

## Verdict for the codebase

- **Adopt:** the dual temper as the standard form of any importance
  correction on regression weights (replaces ad-hoc clamps; interior
  optimum ε≈0.3 here), and the β-trajectory as a standing diagnostic ("is
  my sampling measure moving faster than ε per epoch?").
- **Decline:** trust-region twist scheduling for ssmc (the budget never
  binds — the learned twist is naturally slow; revisit only for ssmc-td's
  bootstrapped targets, the actually-fast-moving object) and concentration
  ramps (harm is ramp-speed-invariant).
- The law-v2 family remains the best known configuration at every
  dimension; probe 4 refines *why*: its sampling measure already moves
  slowly, stays broad, and its only importance corrections are mild.
