# BS=4 winning hyperparameters by dimension

Confirmed-winner hyperparameters (best by 5-seed LCB) for each algorithm, across the original **moons** BS=4 problem and the calibrated random-GMM problem at **d = 2, 8, 32, 128**.  Quad loss throughout.

- *moons d=2* and *GMM d=2* are the **same dimension but different problems** (moons dataset with fixed reward scale s=10 vs. random GMM with the per-dimension gap-calibrated reward).
- `grad_decay = off` means the `use_grad_decay` toggle was False.
- `—` means the hyperparameter is inactive for that config (e.g. `l` only exists when `smc_type = kV_plus_ltr`; `ema_decay` only when `smc_type = k_Vema`).

## off-policy

| hyperparameter | scale | moons d=2 | GMM d=2 | GMM d=8 | GMM d=32 | GMM d=128 |
|---|---|---|---|---|---|---|
| lr | log | 3.57e-04 | 2.81e-04 | 1.04e-04 | 1.07e-04 | 1.07e-04 |
| grad_decay | log (+ on/off toggle) | 4.21e-05 | 1.46e-04 | off | off | off |
| *plateau reward (regret)* | — | — (diff. problem) | -66.26 (reg -65.26) | -7.18 (reg -3.25) | -6.05 (reg -0.69) | -6.07 (reg -0.23) |

## ssmc (single_seed_mc)

| hyperparameter | scale | moons d=2 | GMM d=2 | GMM d=8 | GMM d=32 | GMM d=128 |
|---|---|---|---|---|---|---|
| lr | log | 1.79e-04 | 2.72e-04 | 1.35e-04 | 1.23e-04 | 1.15e-04 |
| grad_decay | log (+ on/off toggle) | off | off | off | off | off |
| n_steps | int (linear) | 24 | 58 | 24 | 21 | 49 |
| mc_samples | log-int | 5 | 2 | 24 | 9 | 2 |
| off_policy_frac | linear [0,.5] | 0.146 | 0.292 | 0.400 | 0.484 | 0.410 |
| smc_type | categorical | k_Vema | kt_r | kt_r | kt_r | k_Vnograd |
| k | log | 5.99e-02 | 3.26e-02 | 2.12e-03 | 3.93e-02 | 2.19e-03 |
| l | log | — | — | — | — | — |
| ema_decay | linear [.9,.999] | 0.905 | — | — | — | — |
| random_t | categorical (bool) | yes | no | no | no | no |
| *plateau reward (regret)* | — | — (diff. problem) | -46.01 (reg -45.01) | -6.44 (reg -2.51) | -6.11 (reg -0.74) | -6.08 (reg -0.24) |

## ssmc-td(λ) (single_seed_td_lambda)

| hyperparameter | scale | moons d=2 | GMM d=2 | GMM d=8 | GMM d=32 | GMM d=128 |
|---|---|---|---|---|---|---|
| lr | log | 1.36e-04 | 1.86e-04 | 1.64e-04 | 1.63e-04 | 1.13e-04 |
| grad_decay | log (+ on/off toggle) | off | 1.06e-05 | off | off | 1.61e-05 |
| n_steps | int (linear) | 22 | 13 | 24 | 17 | 24 |
| mc_samples | log-int | 5 | 13 | 2 | 3 | 1 |
| off_policy_frac | linear [0,.5] | 0.217 | 0.239 | 0.302 | 0.216 | 0.454 |
| smc_type | categorical | k_r | k_Vnograd | k_Vnograd | kV_plus_ltr | k_Vema |
| k | log | 1.16e-01 | 4.32e-01 | 2.44e-02 | 6.24e-01 | 1.93e-01 |
| l | log | — | — | — | 4.31e-02 | — |
| ema_decay | linear [.9,.999] | — | — | — | — | 0.924 |
| random_t | categorical (bool) | yes | yes | no | no | yes |
| lambda_eff | linear [0,1] | 0.171 | 0.461 | 0.811 | 0.623 | 0.632 |
| *plateau reward (regret)* | — | — (diff. problem) | -32.11 (reg -31.11) | -6.47 (reg -2.54) | -6.14 (reg -0.78) | -6.09 (reg -0.25) |

## anc-mc-td(λ) (ancestral_mc_td_lambda)

| hyperparameter | scale | moons d=2 | GMM d=2 | GMM d=8 | GMM d=32 | GMM d=128 |
|---|---|---|---|---|---|---|
| lr | log | 3.71e-04 | 1.17e-04 | 1.13e-04 | 1.33e-04 | 1.39e-04 |
| grad_decay | log (+ on/off toggle) | 4.57e-05 | off | off | off | off |
| n_steps | int (linear) | 10 | 28 | 45 | 30 | 37 |
| mc_samples | log-int | 7 | 4 | 19 | 1 | 17 |
| off_policy_frac | linear [0,.5] | 0.338 | 0.123 | 0.380 | 0.229 | 0.292 |
| smc_type | categorical | kV_plus_ltr | kV_plus_ltr | kt_r | k_Vema | kt_r |
| k | log | 5.72e-02 | 1.44e-03 | 2.78e-02 | 8.56e-02 | 2.87e-03 |
| l | log | 4.62e-03 | 5.58e-01 | — | — | — |
| ema_decay | linear [.9,.999] | — | — | — | 0.955 | — |
| lambda_eff | linear [0,1] | 0.454 | 0.847 | 0.818 | 0.690 | 0.114 |
| *plateau reward (regret)* | — | — (diff. problem) | -28.20 (reg -27.20) | -7.13 (reg -3.20) | -6.13 (reg -0.77) | -6.00 (reg -0.17) |

> Note: the GMM cells and the moons row both use the **discrete `smc_type`** twist space, on the **fixed** estimator. The separate re-tuned moons sweep (`optuna_amctl`) used a more general linear-combination twist (`cr·r + cV·V`), not directly comparable in parameter space, so it is omitted here.
