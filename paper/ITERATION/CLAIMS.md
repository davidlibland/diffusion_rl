# Claim ledger

Every substantive claim in the paper, with the evidence backing it. A claim may
not enter the paper without a row here. Status: **V** verified (proof or
reproducible artifact), **E** empirical (experiment with n and test), **H**
hedged in text as a scope condition, **OPEN** not yet supported.

## Mathematics (S2, S4, App. A, App. D)

| # | Claim | Evidence | Status |
|---|---|---|---|
| M1 | `d/dp D_phi(e^q,e^p) = w(e^p)(e^p-e^q)`, `w=u*phi''` | App. D proof; chain rule | V |
| M2 | Minimiser of `E[D_phi(e^q,e^p)]` is `V` | App. D; classical (Savage/Banerjee) — **attributed** | V |
| M3 | Prediction must be the SECOND Bregman argument | numeric: swapped slots give harmonic mean 0.323 vs `E[T]`=3.08 | V |
| M4 | MSE weight `2u`; explodes above, vanishes below | direct limits; Fig. 1 | V |
| M5 | IS weight `1/u`; bounded `+1` above, explodes below | direct limits; numeric table; Fig. 1 | V |
| M6 | Spence weight `ln u/(u-1)`; both tails linear in `p` | limits; numeric (`p*e^q` matched) | V |
| M7 | No power-law weight gives log-linear tails | sympy: `lim (dL/dp)/p` = inf for `beta>0`, 0 for `beta=0` | V |
| M8 | `phi(u)=(1-u)Li2(1-u) - u ln^2(u)/2` has `phi''=ln u/(u(u-1))` | sympy | V |
| M9 | Value formula equals `D_phi(e^q,e^p)` | sympy | V |
| M10 | Gradient `p(e^p-e^q)/(e^p-1)` is d/dp of the value | sympy | V |
| M11 | `S'(x) = -x e^x/(e^x-1) = x/expm1(-x)` | sympy | V |
| M12 | H-martingale property | App. D proof; numeric on analytic instance | V |
| M13 | HJB residual zero for the analytic instance | numeric: exactly 0 | V |
| M14 | Girsanov twist-independence (full exponential, not one step) | App. D; numeric over 1/4/32 steps, adapted non-optimal twist | V |
| M15 | `KL = E int |u|^2/(4a)` | numeric: 0.82098 vs 0.82134 | V |
| M16 | Backward kernel independent of `nu` | App. D proof; numeric vs GMM base, 3rd moments to 1e-4 | V |
| M17 | Base drift `f = E[(X_1-X_t)/(1-t)|X_t]`, zero iff `nu=N(0,2aI)` | numeric: SDE terminal law matches `gmm_sample` | V |
| M18 | log-space MSE is biased low by the Jensen gap | numeric: minimiser `E[q]`, deficit 1.1246 nats | V |
| M19 | Naive autograd NaNs for target `q >~ 85` (not for large `p`) | numeric sweep; **corrected from an earlier wrong claim** | V |
| M20 | All three losses finite for extreme float32 inputs | numeric grid | V |

## Empirical (S5, S7, App. B)

| # | Claim | Evidence | Status |
|---|---|---|---|
| E1 | **Spence beats exp-MSE** | `experiments/loss_ablation` | **OPEN — running** |
| E2 | IS is an intermediate, better than MSE, worse than Spence | same | **OPEN — running** |
| E3 | log-space MSE converges but to a biased value | same (`v_bias` column) | **OPEN — running** |
| E4 | On-policy beats off-policy at `d=8..32` | n=30 paired, `p<0.05`; `results.json` | E |
| E5 | Naive on-policy laws collapse at high `d` (-7.8 at 512) | n=30 paired, `p<0.05` | E |
| E6 | Refitted laws match/beat off-policy at every `d` in 2..512 | n=30 paired; ties elsewhere, never sig. below | E |
| E7 | Expansion + finer integration stack (+7.0) | n=10 paired, App. B | E |
| E8 | Staleness explains none of the gain | `_sub` arms match within noise | E |
| E9 | Concentration harms at `d=512` even with oracle twist | App. B | E |
| E10 | Trust region inert for the learned twist | beta=1.0 at all ~50 updates | E |
| E11 | GMM-40 log-Z bias 0.08, ahead of published | 3 seeds; **n is small — hedge** | E/H |
| E12 | Many-Well-32 not competitive; cause diagnosed | reported as held-out negative | E |

## Scope conditions carried in the text (S9)

| # | Condition | Where |
|---|---|---|
| H1 | Rewards are cheap; we count samples, not reward calls | S2.4, S9 |
| H2 | On-policy recipe does not transfer to Boltzmann targets | S6, S7.3, S9 |
| H3 | Headroom fixed at 6 nats; width capped at 256 | S9 |
| H4 | GMM-40 sample metric not comparable to published `W2` | S7.3 |
| H5 | Loss ablation is off-policy only, so it isolates the divergence but says nothing about loss x sampler interaction | S5 — **must be stated when written** |
