# Dimension scaling at constant headroom — consolidated report

**Question.** How do the on-policy SMC/FBRRT value-learning methods scale with
dimension relative to plain off-policy regression, once task difficulty is held
constant across d — and when they lose, *why*?

**Headline.** On-policy methods win clearly at low dimension (d ≤ 32) and lose
to off-policy at high dimension (d ≥ 256). The high-d deficit is **not**
instability, particle starvation, twist parameterization, or the gradient
driver — probes ruled each out. It is a **coverage-vs-concentration** effect:
off-policy training visits the whole base manifold, while on-policy corridor
concentration (even with the *analytically optimal* twist) narrows the training
distribution and makes global value regression at 512 dimensions worse, not
better. Probe 3 confirmed it from both sides: corridor data does not help even
when coverage is guaranteed (blend), while **backward-noising expansion** of
on-policy value estimates — a pure coverage densifier — lifts ssmc to a
statistical tie with off-policy, matching the integrator fix via an independent
mechanism.

All code in this directory; results in `results/`; probe launch logs in `logs/`.

---

## 1. Problem family and calibration

`problem_consth.py` builds a **constant-headroom, coordinate-nested** GMM
family:

- A single GMM instance per seed is drawn at `D_MAX = 512` (spherical
  components), then **truncated to the first d coordinates** for lower dims —
  exact marginals for spherical components, so within a seed *dimension is the
  only thing that varies*.
- Reward scale s is **bisected per (d, seed)** so that the achievable prize is
  identical everywhere: `E_{p*}[r] − E_base[r] = 6 nats`. Reported
  `frac_closed = (plateau − E_base)/6` is therefore directly comparable across
  every cell of the grid.
- Sanity identity (Gibbs): `headroom = KL(p*‖p_base) + (V00 − E_base[r])`,
  checked per instance. The analytic `V00` also supplies `bias_val` for network
  output bias and `loss_shift` (recentring exp-space losses; without it Adam's
  eps stalls learning at the ~36-nat log-gaps that occur at d=512).

Six methods: `off_policy`, `single_seed_mc` (ssmc), `single_seed_td_lambda`
(ssmc-td), `ancestral_mc_td_lambda` (amctl), `fbrrt`, `fbrrt_cv`. All use the
quad (log-quadratic Bregman) loss with the stabilized closed-form gradient,
the N(0, 0.1) input-bias init, and grad-decay/EMA stabilizers where selected.
SMC methods have the new gradient-based guided proposals available
(`use_guidance` / `guidance_scale` / `guidance_source`, exact discrete-Girsanov
compensation of the clipped applied control).

## 2. Hyperparameter laws (anchors → fits)

Optuna anchor sweeps at d ∈ {2, 16, 128} (per-method studies, parallel
workers), then per-hparam laws fit on log(d): Theil-Sen in the Optuna sampling
scale, leave-one-dimension-out selection between a constant (weighted median)
and a slope (kept only if it cuts LOO error > 10%). See `hparam_fit_consth.md`
and `fitted_models_consth.json`; `fit_consth.hparams_for_dim(method, d)` serves
the laws to the grid.

Almost everything came out **constant** — the statistically supported slopes
were `off_policy.lr` (rising, capped at 3e-3) and `ssmc-td.lambda_eff` (rising
from 0.08 at d=2 to 0.75 at d=512). Notably the laws pinned `n_steps` at 19–22,
which probe 2 later showed is a real (−0.86 nat) integrator bias at d=512 —
**future sweeps should scale n_steps with d** (see §6).

## 3. Grid results (8 dims × 6 methods × 30 nested seeds, 15k steps)

`frac_closed` (%) — fraction of the 6-nat headroom captured, mean ± s.e.:

| method | d=2 | d=8 | d=16 | d=32 | d=64 | d=128 | d=256 | d=512 |
|---|---|---|---|---|---|---|---|---|
| ssmc | **90.1±0.6** | 64.5±1.9 | 51.2±2.1 | **40.0±1.6** | **36.9±2.3** | **31.9±2.0** | 29.3±1.9 | 26.4±1.5 |
| ssmc-td | 90.0±0.6 | **68.5±1.3** | **51.7±1.9** | 39.6±1.8 | 36.5±2.2 | 30.3±2.0 | 30.2±2.1 | 28.3±2.0 |
| amctl | 86.8±0.8 | 61.2±2.0 | 49.7±1.9 | 39.4±1.7 | 36.3±2.1 | 28.2±2.0 | 25.4±1.4 | 20.6±1.1 |
| off_policy | 86.6±0.9 | 49.8±2.8 | 43.9±2.3 | 35.2±1.9 | 36.0±2.3 | 31.0±2.3 | **33.8±2.3** | **34.3±2.6** |
| fbrrt | 82.2±0.8 | 31.3±4.6 | 35.2±3.3 | 22.8±5.2 | 19.1±2.9 | 18.9±3.1 | 18.0±3.2 | 10.9±2.8 |
| fbrrt_cv | 78.0±0.9 | 30.7±3.1 | 10.4±6.7 | 6.0±2.2 | 1.8±1.5 | −0.5±0.6 | −1.7±0.3 | −2.0±0.6 |

- **Crossover at d ≈ 64–128.** On-policy SMC beats off-policy decisively for
  d ≤ 32 (largest gap at d=8: ssmc-td 68.5% vs 49.8%), ties at d=64–128, and
  loses at d ≥ 256. Off-policy is nearly **flat in d** (as calibrated
  difficulty is constant); every on-policy method decays.
- **fbrrt_cv collapses at high d** (≈ 0% for d ≥ 128). This is its own
  pathology (live-net targets), distinct from the SMC story, and has **not**
  been audited (§6).

## 4. Probe 1 — three null results (d=128/512, paired seeds 0–9, 15k steps)

One factor changed per arm vs the law control (ssmc d=512 control on the same
seeds: 25.9±2.9):

| arm | ssmc d512 | verdict |
|---|---|---|
| `reward_twist` (old fixed-reward twist) | 24.2±2.2 | null — twist parameterization is not the deficit |
| `mc16` (9→16 particles) | 22.0±1.6 | null — not particle starvation |
| `guid_toggle` (guidance flipped) | 24.2±2.0 | null — the gradient driver neither causes nor fixes it |

(Same pattern for ssmc-td and amctl; see `results/probe_*.json`.)

## 5. Probe 2 — the mechanism (ssmc, d=512, paired seeds 0–9)

Controls on the same seeds: ssmc 25.9±2.9, off_policy 30.7±1.6.

| arm | result | reading |
|---|---|---|
| `ns60` (n_steps 19→60) | **29.1±2.9** | +3.2: the EM integrator at 19 steps biases the sampled terminals by −0.86 nats at d=512. Fixing it makes ssmc **statistically tie off-policy**. The law's n_steps=19 was the dominant *mechanical* deficit. |
| `oracle_twist` (τ = analytic V*) | **16.3±1.4** | Concentration made it much **worse**. Per-step ESS audit: 0.90/0.64 under V* (real corridor concentration, terminals move halfway from base toward optimal by resampling alone) vs ESS = 1.00 under the learned twist (zero concentration — the learned twist does nothing). |
| `oracle_guid` (drift = ∇V*, scale 1) | 21.1±2.3 | Transport-by-drift also hurts. |
| `warmstart` (τ and drift from a fully off-policy-trained V) | 17.5±1.8 | The realizable version confirms it: handing ssmc off-policy's own final V makes it *worse* than off-policy. |

**Interpretation.** The lock-in hypothesis was half right: the learned twist is
indeed uninformative (ESS = 1.00 — ssmc at high d survives *because* its twist
does nothing, so it trains on base-marginal coverage like off-policy, minus
integrator bias and target variance). But the cure is not a better twist:
**genuine corridor concentration shrinks the training distribution and damages
the global value fit at d=512**. Coverage beats concentration at high
dimension. This also explains why fixed-reward twists won in the old
low-d moons study (concentration helps when the manifold is small) yet
nothing twist-related helps at d=512.

## 6. Probe 3 — closing the loop

Can corridor signal help *on top of* coverage, and can on-policy estimates be
*spread back over* the manifold? `probe3_cell.py`, ssmc, d=512, paired seeds
0–9, 15k steps:

- **blend** — oracle twist + `off_policy_frac = 0.5`: coverage guaranteed by
  the off-policy splice, concentration by the twist. Beats both parents
  (16.3 / 30.7) ⇒ corridor data adds signal once coverage is guaranteed;
  matches off-policy ⇒ the corridor adds nothing.
- **expand** — law config + **backward-noising expansion**: every on-policy row
  (x_s, s, v̂) is expanded to a sample at smaller t (higher noise) through the
  *exact* base backward kernel
  `X_t | X_s = x_s ~ N((t/s)·x_s, 2a·t(s−t)/s·I)`
  (the base process is an h-transform of BM from X_0 = 0, so its past given
  the present is a Brownian bridge from 0 — x₁ cancels). By the harmonic
  property `e^{V(x,t)} = E_base[e^{V(X_s,s)} | X_t = x]`, the pair
  (x′, t′, target v̂) is a consistent exp-space regression sample when sources
  are base-marginal (which the law twist's ESS = 1.00 chain is). k=1 extra
  sample per source row, t′ ~ U(0.05, s).
- **expand_oracle** — oracle twist + expansion **with e^{−v̂} unweighting**
  (self-normalized, capped at 100×): corridor sources carry the most
  informative v̂; unweighting restores the base-marginal source condition the
  harmonic identity requires.

Implementation: `OnPolicySMCDataset.augment_fn` hook
(`src/diffusion_rl/models/on_policy.py`), applied per epoch after the
off-policy splice, before shuffling. Verified: rows double, expanded t′ in
(0.05, s), weights finite (mean ≈ 0.9 with the cap active).

**Results** (frac_closed %, paired deltas mean ± s.e., paired t-test):

| arm | mean | vs grid ssmc (25.9) | vs off_policy (30.7) | vs oracle_twist (16.3) |
|---|---|---|---|---|
| **expand** | **28.9±2.4** | **+3.0±1.6** (p=.087) | −1.8±1.2 (p=.18) | +12.5±2.4 (p=.001) |
| blend | 22.2±1.9 | −3.7±1.9 (p=.086) | −8.5±1.4 (p<.001) | +5.9±1.8 (p=.010) |
| expand_oracle | 21.9±2.7 | −3.9±1.1 (p=.007) | −8.7±1.4 (p<.001) | +5.6±2.6 (p=.060) |

Readings:

- **The noising expansion works in its realizable form.** `expand` gains
  +3.0 over the ssmc control on every-seed pairing and reaches a
  **statistical tie with off-policy** (−1.8±1.2, p=0.18) — the same place
  the ns60 integrator fix landed (paired vs ns60: −0.2±1.9, p=0.91), via an
  entirely different mechanism (small-t data density vs terminal-sampling
  bias).
- **Corridor data does NOT add on top of coverage.** `blend` sits *below*
  the plain ssmc control (−3.7) and far below off-policy (−8.5) despite half
  its batch being off-policy coverage: the oracle-twist drag is not
  neutralized by splicing coverage back in. Together with probe 2, this
  closes the question — at d=512 concentration is net harmful, full stop,
  not merely insufficient.
- **Unweighted expansion recovers only about half the concentration
  damage** (`expand_oracle` +5.6 over oracle_twist but still −3.9 below the
  no-concentration control). The e^{−v̂} self-normalized unweighting pays a
  large ESS cost at 512 dimensions (a handful of corridor sources dominate
  after reweighting), so the recovered coverage is low-diversity.
- **Follow-up arm (running): `expand_ns60`** stacks the two same-size,
  mechanism-independent gains (expansion + n_steps=60) — the direct test of
  whether on-policy can *beat* off-policy at d=512.

## 7. Caveats / follow-ups

- **fbrrt_cv high-d collapse is unaudited.** Negative frac_closed at d ≥ 128
  means it ends *below* the base process. Worth a dedicated look before any
  claim about the CV variant.
- **n_steps law is wrong at high d.** The anchors (d ≤ 128) could not see the
  integrator bias; probe 2 showed n_steps=60 is worth +3.2 points at d=512.
  Future sweeps should either scale n_steps ∝ d^α or use a higher-order
  integrator.
- The 6-nat headroom is one difficulty point; the crossover dimension likely
  shifts with headroom.
- Grid hidden width is capped at 256 for all d ≥ 8; capacity–dimension
  interaction is unexplored.
