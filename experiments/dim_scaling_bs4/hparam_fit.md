# Hyperparameter-vs-dimension fits (BS=4, quad loss)

Per algorithm, each hyperparameter is fit as a function of **log(dimension)** from the **top-3** Optuna configs at moons d=2 + GMM d∈{2,8,32,128}, rank-weighted 3/2/1 within each setting.  Continuous hparams use the Optuna sampling-scale transform (log / logit / linear) and a robust Theil-Sen fit; a **slope** is kept over a **constant** only if it lowers leave-one-dimension-out error by >10% (else the hparam is flagged *noise* and set to a dimension-independent constant).  Categoricals/booleans use the rank-weighted mode/majority.

## off_policy

| hparam | scale | model | noise? | LOO const→slope | d=2 | d=8 | d=16 | d=32 | d=64 | d=128 |
|---|---|---|---|---|---|---|---|---|---|---|
| lr | log | const | noise | 1.61→2.24 | 1.13e-04 | 1.13e-04 | 1.13e-04 | 1.13e-04 | 1.13e-04 | 1.13e-04 |
| use_grad_decay | bool | bool | — | — | off | off | off | off | off | off |
| grad_decay | log | const | noise | —(const) | — | — | — | — | — | — |

## single_seed_mc

| hparam | scale | model | noise? | LOO const→slope | d=2 | d=8 | d=16 | d=32 | d=64 | d=128 |
|---|---|---|---|---|---|---|---|---|---|---|
| lr | log | const | noise | 1.12→1.09 | 1.75e-04 | 1.75e-04 | 1.75e-04 | 1.75e-04 | 1.75e-04 | 1.75e-04 |
| n_steps | cont | const | noise | 239.52→407.35 | 37 | 37 | 37 | 37 | 37 | 37 |
| mc_samples | log | const | noise | 2.20→3.71 | 2 | 2 | 2 | 2 | 2 | 2 |
| off_policy_frac | cont | const | noise | 4.15→6.42 | 0.309 | 0.309 | 0.309 | 0.309 | 0.309 | 0.309 |
| k | log | slope | trend | 7.31→6.19 | 9.26e-02 | 3.93e-02 | 2.56e-02 | 1.67e-02 | 1.09e-02 | 7.10e-03 |
| smc_type | cat | cat | — | — | kt_r | kt_r | kt_r | kt_r | kt_r | kt_r |
| use_grad_decay | bool | bool | — | — | off | off | off | off | off | off |
| random_t | bool | bool | — | — | False | False | False | False | False | False |
| grad_decay | log | const | noise | —(const) | — | — | — | — | — | — |
| ema_decay | cont | const | noise | —(const) | — | — | — | — | — | — |

## single_seed_td_lambda

| hparam | scale | model | noise? | LOO const→slope | d=2 | d=8 | d=16 | d=32 | d=64 | d=128 |
|---|---|---|---|---|---|---|---|---|---|---|
| lr | log | const | noise | 0.37→0.51 | 1.64e-04 | 1.64e-04 | 1.64e-04 | 1.64e-04 | 1.64e-04 | 1.64e-04 |
| n_steps | cont | const | noise | 78.38→163.03 | 22 | 22 | 22 | 22 | 22 | 22 |
| mc_samples | log | const | noise | 1.20→1.14 | 4 | 4 | 4 | 4 | 4 | 4 |
| off_policy_frac | cont | slope | trend | 2.18→1.86 | 0.198 | 0.296 | 0.341 | 0.381 | 0.413 | 0.438 |
| k | log | const | noise | 1.65→4.47 | 1.74e-01 | 1.74e-01 | 1.74e-01 | 1.74e-01 | 1.74e-01 | 1.74e-01 |
| lambda_eff | cont | slope | trend | 6.43→4.88 | 0.328 | 0.551 | 0.661 | 0.756 | 0.831 | 0.886 |
| smc_type | cat | cat | — | — | k_r | k_r | k_r | k_r | k_r | k_r |
| use_grad_decay | bool | bool | — | — | off | off | off | off | off | off |
| random_t | bool | bool | — | — | True | True | True | True | True | True |
| grad_decay | log | const | noise | 1.83→27.28 | — | — | — | — | — | — |
| l | log | const | noise | —(const) | — | — | — | — | — | — |
| ema_decay | cont | const | noise | —(const) | — | — | — | — | — | — |

## ancestral_mc_td_lambda

| hparam | scale | model | noise? | LOO const→slope | d=2 | d=8 | d=16 | d=32 | d=64 | d=128 |
|---|---|---|---|---|---|---|---|---|---|---|
| lr | log | const | noise | 0.83→1.14 | 1.37e-04 | 1.37e-04 | 1.37e-04 | 1.37e-04 | 1.37e-04 | 1.37e-04 |
| n_steps | cont | const | noise | 288.22→702.32 | 32 | 32 | 32 | 32 | 32 | 32 |
| mc_samples | log | const | noise | 1.87→3.85 | 3 | 3 | 3 | 3 | 3 | 3 |
| off_policy_frac | cont | const | noise | 1.01→1.35 | 0.340 | 0.340 | 0.340 | 0.340 | 0.340 | 0.340 |
| k | log | const | noise | 5.49→13.64 | 1.67e-02 | 1.67e-02 | 1.67e-02 | 1.67e-02 | 1.67e-02 | 1.67e-02 |
| lambda_eff | cont | const | noise | 4.36→9.05 | 0.714 | 0.714 | 0.714 | 0.714 | 0.714 | 0.714 |
| smc_type | cat | cat | — | — | k_Vnograd | k_Vnograd | k_Vnograd | k_Vnograd | k_Vnograd | k_Vnograd |
| use_grad_decay | bool | bool | — | — | off | off | off | off | off | off |
| grad_decay | log | const | noise | —(const) | — | — | — | — | — | — |
| l | log | const | noise | —(const) | — | — | — | — | — | — |
| ema_decay | cont | const | trend | 0.27→0.06 | — | — | — | — | — | — |
