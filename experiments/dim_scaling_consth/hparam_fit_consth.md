# Hyperparameter-vs-dimension fits (constant-headroom family)


## off_policy  (3 records)

| hparam | model | LOO const→slope | d=2 | d=16 | d=128 | d=512 |
|---|---|---|---|---|---|---|
| lr | const | — | 0.0002465 | 0.0002465 | 0.0002465 | 0.0002465 |
| use_grad_decay | bool (p=1.00) | — | True ||||
| grad_decay | const | — | 0.02915 | 0.02915 | 0.02915 | 0.02915 |

## single_seed_mc  (3 records)

| hparam | model | LOO const→slope | d=2 | d=16 | d=128 | d=512 |
|---|---|---|---|---|---|---|
| lr | const | — | 0.0003941 | 0.0003941 | 0.0003941 | 0.0003941 |
| n_steps | const | — | 17 | 17 | 17 | 17 |
| mc_samples | const | — | 2 | 2 | 2 | 2 |
| off_policy_frac | const | — | 0.1082 | 0.1082 | 0.1082 | 0.1082 |
| k | const | — | 0.002223 | 0.002223 | 0.002223 | 0.002223 |
| guidance_scale | const | — | 0.06238 | 0.06238 | 0.06238 | 0.06238 |
| smc_type | cat | — | k_r ||||
| guidance_source | cat | — | live ||||
| use_grad_decay | bool (p=0.50) | — | True ||||
| use_guidance | bool (p=0.83) | — | True ||||
| random_t | bool (p=0.50) | — | True ||||
| grad_decay | const | — | 3.077e-05 | 3.077e-05 | 3.077e-05 | 3.077e-05 |
| l | const | — | 0.004335 | 0.004335 | 0.004335 | 0.004335 |
| ema_decay | const | — | 0.9913 | 0.9913 | 0.9913 | 0.9913 |

## single_seed_td_lambda  (3 records)

| hparam | model | LOO const→slope | d=2 | d=16 | d=128 | d=512 |
|---|---|---|---|---|---|---|
| lr | const | — | 0.0003575 | 0.0003575 | 0.0003575 | 0.0003575 |
| n_steps | const | — | 17 | 17 | 17 | 17 |
| mc_samples | const | — | 1 | 1 | 1 | 1 |
| off_policy_frac | const | — | 0.1412 | 0.1412 | 0.1412 | 0.1412 |
| k | const | — | 0.3143 | 0.3143 | 0.3143 | 0.3143 |
| lambda_eff | const | — | 0.4319 | 0.4319 | 0.4319 | 0.4319 |
| smc_type | cat | — | kV_plus_ltr ||||
| use_grad_decay | bool (p=1.00) | — | True ||||
| use_guidance | bool (p=0.00) | — | False ||||
| random_t | bool (p=0.50) | — | True ||||
| grad_decay | const | — | 0.001541 | 0.001541 | 0.001541 | 0.001541 |
| l | const | — | 0.004335 | 0.004335 | 0.004335 | 0.004335 |

## ancestral_mc_td_lambda  (3 records)

| hparam | model | LOO const→slope | d=2 | d=16 | d=128 | d=512 |
|---|---|---|---|---|---|---|
| lr | const | — | 0.0003575 | 0.0003575 | 0.0003575 | 0.0003575 |
| n_steps | const | — | 17 | 17 | 17 | 17 |
| mc_samples | const | — | 1 | 1 | 1 | 1 |
| off_policy_frac | const | — | 0.078 | 0.078 | 0.078 | 0.078 |
| k | const | — | 0.1068 | 0.1068 | 0.1068 | 0.1068 |
| guidance_scale | const | — | 0.06238 | 0.06238 | 0.06238 | 0.06238 |
| lambda_eff | const | — | 0.576 | 0.576 | 0.576 | 0.576 |
| smc_type | cat | — | kV_plus_ltr ||||
| guidance_source | cat | — | live ||||
| use_grad_decay | bool (p=0.83) | — | True ||||
| use_guidance | bool (p=0.50) | — | True ||||
| grad_decay | const | — | 0.002481 | 0.002481 | 0.002481 | 0.002481 |
| l | const | — | 0.004335 | 0.004335 | 0.004335 | 0.004335 |

## fbrrt  (3 records)

| hparam | model | LOO const→slope | d=2 | d=16 | d=128 | d=512 |
|---|---|---|---|---|---|---|
| lr | const | — | 0.001445 | 0.001445 | 0.001445 | 0.001445 |
| n_steps | const | — | 32 | 32 | 32 | 32 |
| mc_samples | const | — | 4 | 4 | 4 | 4 |
| branch | const | — | 2 | 2 | 2 | 2 |
| alpha | const | — | 0.4996 | 0.4996 | 0.4996 | 0.4996 |
| off_policy_frac | const | — | 0.04333 | 0.04333 | 0.04333 | 0.04333 |
| entropy_lambda | const | — | 2.069 | 2.069 | 2.069 | 2.069 |
| ema_decay | const | — | 0.9381 | 0.9381 | 0.9381 | 0.9381 |
| use_grad_decay | bool (p=0.50) | — | True ||||
| ent_inf | bool (p=0.17) | — | False ||||
| grad_decay | const | — | 5.415e-05 | 5.415e-05 | 5.415e-05 | 5.415e-05 |

## fbrrt_cv  (3 records)

| hparam | model | LOO const→slope | d=2 | d=16 | d=128 | d=512 |
|---|---|---|---|---|---|---|
| lr | const | — | 0.001445 | 0.001445 | 0.001445 | 0.001445 |
| n_steps | const | — | 32 | 32 | 32 | 32 |
| mc_samples | const | — | 4 | 4 | 4 | 4 |
| branch | const | — | 2 | 2 | 2 | 2 |
| alpha | const | — | 0.4996 | 0.4996 | 0.4996 | 0.4996 |
| off_policy_frac | const | — | 0.04333 | 0.04333 | 0.04333 | 0.04333 |
| entropy_lambda | const | — | 2.069 | 2.069 | 2.069 | 2.069 |
| ema_decay | const | — | 0.9381 | 0.9381 | 0.9381 | 0.9381 |
| use_grad_decay | bool (p=0.50) | — | True ||||
| ent_inf | bool (p=0.17) | — | False ||||
| grad_decay | const | — | 5.415e-05 | 5.415e-05 | 5.415e-05 | 5.415e-05 |
