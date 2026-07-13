# Law-v2 hyperparameter-vs-dimension fits (anchors 2/16/128/512)


## single_seed_mc  (12 records)

| hparam | model | LOO const→slope | d=2 | d=16 | d=128 | d=512 |
|---|---|---|---|---|---|---|
| lr | slope | 3.61→0.62 | 0.0001 | 0.0002548 | 0.0007215 | 0.001444 |
| n_steps | const | 0.67→1.69 | 29 | 29 | 29 | 29 |
| mc_samples | const | 2.16→7.89 | 5 | 5 | 5 | 5 |
| off_policy_frac | const | 2.62→3.72 | 0.1747 | 0.1747 | 0.1747 | 0.1747 |
| k | const | 11.26→12.39 | 0.1345 | 0.1345 | 0.1345 | 0.1345 |
| guidance_scale | const | 0.31→0.63 | 0.1317 | 0.1317 | 0.1317 | 0.1317 |
| expand_frac | const | 1.70→2.31 | 0.4561 | 0.4561 | 0.4561 | 0.4561 |
| epoch_rows | const | 2.00→4.15 | 4594 | 4594 | 4594 | 4594 |
| smc_type | cat | — | k_Vema ||||
| guidance_source | cat | — | live ||||
| use_grad_decay | bool (p=0.83) | — | True ||||
| use_guidance | bool (p=0.46) | — | False ||||
| random_t | bool (p=0.46) | — | False ||||
| grad_decay | const | — | 0.0006632 | 0.0006632 | 0.0006632 | 0.0006632 |
| l | const | — | 0.03878 | 0.03878 | 0.03878 | 0.03878 |
| ema_decay | const | — | 0.9768 | 0.9768 | 0.9768 | 0.9768 |

## single_seed_td_lambda  (12 records)

| hparam | model | LOO const→slope | d=2 | d=16 | d=128 | d=512 |
|---|---|---|---|---|---|---|
| lr | slope | 4.94→1.00 | 0.0001 | 0.0001644 | 0.0004516 | 0.0008859 |
| n_steps | const | 0.50→1.36 | 33 | 33 | 33 | 33 |
| mc_samples | slope | 1.65→1.37 | 3 | 2 | 2 | 1 |
| off_policy_frac | const | 2.54→5.24 | 0.1879 | 0.1879 | 0.1879 | 0.1879 |
| k | const | 4.60→6.26 | 0.04123 | 0.04123 | 0.04123 | 0.04123 |
| guidance_scale | const | 0.52→3.08 | 0.5125 | 0.5125 | 0.5125 | 0.5125 |
| expand_frac | const | 3.21→7.53 | 0.6875 | 0.6875 | 0.6875 | 0.6875 |
| epoch_rows | const | 0.47→1.04 | 1359 | 1359 | 1359 | 1359 |
| lambda_eff | const | 2.49→5.04 | 0.3172 | 0.3172 | 0.3172 | 0.3172 |
| smc_type | cat | — | kt_r ||||
| guidance_source | cat | — | ema ||||
| use_grad_decay | bool (p=0.38) | — | False ||||
| use_guidance | bool (p=0.50) | — | True ||||
| random_t | bool (p=0.33) | — | False ||||
| grad_decay | const | — | 0.0001627 | 0.0001627 | 0.0001627 | 0.0001627 |
| l | const | — | 0.5022 | 0.5022 | 0.5022 | 0.5022 |
| ema_decay | const | — | 0.9436 | 0.9436 | 0.9436 | 0.9436 |
