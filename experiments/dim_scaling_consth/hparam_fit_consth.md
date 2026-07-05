# Hyperparameter-vs-dimension fits (constant-headroom family)


## off_policy  (9 records)

| hparam | model | LOO const→slope | d=2 | d=16 | d=128 | d=512 |
|---|---|---|---|---|---|---|
| lr | slope | 4.15→0.51 | 0.0001754 | 0.0007788 | 0.003 | 0.003 |
| use_grad_decay | bool (p=0.72) | — | True ||||
| grad_decay | const | — | 0.000173 | 0.000173 | 0.000173 | 0.000173 |

## single_seed_mc  (9 records)

| hparam | model | LOO const→slope | d=2 | d=16 | d=128 | d=512 |
|---|---|---|---|---|---|---|
| lr | const | 1.59→2.61 | 0.0004196 | 0.0004196 | 0.0004196 | 0.0004196 |
| n_steps | const | 381.25→1365.25 | 19 | 19 | 19 | 19 |
| mc_samples | const | 2.52→2.86 | 9 | 9 | 9 | 9 |
| off_policy_frac | const | 4.95→14.98 | 0.1669 | 0.1669 | 0.1669 | 0.1669 |
| k | const | 3.29→10.61 | 0.196 | 0.196 | 0.196 | 0.196 |
| guidance_scale | const | 1.58→2.96 | 0.6091 | 0.6091 | 0.6091 | 0.6091 |
| smc_type | cat | — | k_Vema ||||
| guidance_source | cat | — | ema ||||
| use_grad_decay | bool (p=1.00) | — | True ||||
| use_guidance | bool (p=0.67) | — | True ||||
| random_t | bool (p=0.33) | — | False ||||
| grad_decay | const | — | 0.0001468 | 0.0001468 | 0.0001468 | 0.0001468 |
| l | const | — | 0.1553 | 0.1553 | 0.1553 | 0.1553 |
| ema_decay | const | — | 0.9623 | 0.9623 | 0.9623 | 0.9623 |

## single_seed_td_lambda  (9 records)

| hparam | model | LOO const→slope | d=2 | d=16 | d=128 | d=512 |
|---|---|---|---|---|---|---|
| lr | const | 0.16→0.37 | 0.0001678 | 0.0001678 | 0.0001678 | 0.0001678 |
| n_steps | const | 41.69→94.22 | 19 | 19 | 19 | 19 |
| mc_samples | const | 1.72→3.83 | 9 | 9 | 9 | 9 |
| off_policy_frac | const | 0.33→0.62 | 0.1943 | 0.1943 | 0.1943 | 0.1943 |
| k | const | 8.74→8.39 | 0.005088 | 0.005088 | 0.005088 | 0.005088 |
| guidance_scale | const | — | 0.05095 | 0.05095 | 0.05095 | 0.05095 |
| lambda_eff | slope | 3.70→1.17 | 0.08063 | 0.2483 | 0.5544 | 0.7508 |
| smc_type | cat | — | kV_plus_ltr ||||
| guidance_source | cat | — | ema ||||
| use_grad_decay | bool (p=0.83) | — | True ||||
| use_guidance | bool (p=0.17) | — | False ||||
| random_t | bool (p=0.67) | — | True ||||
| grad_decay | const | — | 0.0004244 | 0.0004244 | 0.0004244 | 0.0004244 |
| l | const | — | 0.0392 | 0.0392 | 0.0392 | 0.0392 |
| ema_decay | const | — | 0.9532 | 0.9532 | 0.9532 | 0.9532 |

## ancestral_mc_td_lambda  (9 records)

| hparam | model | LOO const→slope | d=2 | d=16 | d=128 | d=512 |
|---|---|---|---|---|---|---|
| lr | const | 1.01→1.13 | 0.0002907 | 0.0002907 | 0.0002907 | 0.0002907 |
| n_steps | const | 273.06→884.06 | 22 | 22 | 22 | 22 |
| mc_samples | const | 0.00→0.00 | 1 | 1 | 1 | 1 |
| off_policy_frac | const | 4.36→4.06 | 0.119 | 0.119 | 0.119 | 0.119 |
| k | const | 3.15→14.73 | 0.003315 | 0.003315 | 0.003315 | 0.003315 |
| guidance_scale | const | 1.19→2.68 | 0.06496 | 0.06496 | 0.06496 | 0.06496 |
| lambda_eff | const | 1.35→6.85 | 0.5987 | 0.5987 | 0.5987 | 0.5987 |
| smc_type | cat | — | kV_plus_ltr ||||
| guidance_source | cat | — | live ||||
| use_grad_decay | bool (p=0.72) | — | True ||||
| use_guidance | bool (p=0.78) | — | True ||||
| grad_decay | const | — | 0.001233 | 0.001233 | 0.001233 | 0.001233 |
| l | const | — | 0.01703 | 0.01703 | 0.01703 | 0.01703 |
| ema_decay | const | — | 0.9538 | 0.9538 | 0.9538 | 0.9538 |

## fbrrt  (9 records)

| hparam | model | LOO const→slope | d=2 | d=16 | d=128 | d=512 |
|---|---|---|---|---|---|---|
| lr | const | 2.05→6.81 | 0.0002547 | 0.0002547 | 0.0002547 | 0.0002547 |
| n_steps | const | 906.08→2774.79 | 30 | 30 | 30 | 30 |
| mc_samples | const | 3.45→12.70 | 5 | 5 | 5 | 5 |
| branch | const | 0.25→0.67 | 12 | 12 | 12 | 12 |
| alpha | const | 0.31→0.94 | 0.4941 | 0.4941 | 0.4941 | 0.4941 |
| off_policy_frac | const | 14.85→78.60 | 0.1256 | 0.1256 | 0.1256 | 0.1256 |
| entropy_lambda | const | — | 1.217 | 1.217 | 1.217 | 1.217 |
| ema_decay | const | 2.76→12.94 | 0.9842 | 0.9842 | 0.9842 | 0.9842 |
| use_grad_decay | bool (p=0.72) | — | True ||||
| ent_inf | bool (p=0.61) | — | True ||||
| grad_decay | const | — | 0.001871 | 0.001871 | 0.001871 | 0.001871 |

## fbrrt_cv  (9 records)

| hparam | model | LOO const→slope | d=2 | d=16 | d=128 | d=512 |
|---|---|---|---|---|---|---|
| lr | const | 0.58→2.28 | 0.0002651 | 0.0002651 | 0.0002651 | 0.0002651 |
| n_steps | const | 666.42→3109.67 | 31 | 31 | 31 | 31 |
| mc_samples | const | 2.76→13.82 | 7 | 7 | 7 | 7 |
| branch | const | 1.25→1.76 | 3 | 3 | 3 | 3 |
| alpha | const | 0.31→1.07 | 0.7671 | 0.7671 | 0.7671 | 0.7671 |
| off_policy_frac | const | 8.11→41.81 | 0.05333 | 0.05333 | 0.05333 | 0.05333 |
| entropy_lambda | const | — | 0.62 | 0.62 | 0.62 | 0.62 |
| ema_decay | const | 1.10→2.05 | 0.9412 | 0.9412 | 0.9412 | 0.9412 |
| use_grad_decay | bool (p=0.72) | — | True ||||
| ent_inf | bool (p=0.61) | — | True ||||
| grad_decay | const | — | 0.0002766 | 0.0002766 | 0.0002766 | 0.0002766 |
