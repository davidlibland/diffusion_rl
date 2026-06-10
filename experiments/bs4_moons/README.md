# bs4_moons — on-policy vs off-policy at gradient batch size 4

Benchmark comparing all on-policy sampling methods and the off-policy baseline
on the moons/GMM task at BS=4 (Optuna sweep → 5-seed confirm → 50k-step
convergence; detrended-SEM LCB objective). Runs live in date-marked folders:

| folder | what | code |
|---|---|---|
| [`2026-05-28/`](2026-05-28/) | Original full study (all 10 methods; summary in [`bs4_experiments_summary.md`](2026-05-28/bs4_experiments_summary.md)). Includes the 2026-05-29 `ancestral_mc_td_lambda` re-tune. **⚠️ The FBRRT results predate the FBRRT FBSDE fixes** (commits `57926d6`, `13ef36a`, `c6b70ee`: entropy weights moved off the target, ancestor-aligned TD(λ), corrected CV driver/Malliavin scaling, child-time driver gradients). The FBRRT rows there (`fbrrt_td_lambda` plateau −10.71, "weakest and slowest") reflect the **old, biased estimators** and should not be cited as the methods' performance. Non-FBRRT rows are unaffected. | pre-fix (≤ `3ec4f7b`) |
| [`2026-06-10/`](2026-06-10/) | **FBRRT re-tune** under the fixed estimators, same task/objective/budgets, to test whether FBRRT can match the other on-policy methods and the off-policy baseline. | post-fix (`c6b70ee`+) |

Reference numbers from the archived study (plateau reward at 50k, BS=4):
`single_seed_td_lambda` **−5.16** · `ancestral_mc_td_lambda` (re-tuned) **−5.67**
· `single_seed_mc` **−6.78** · off-policy **−9.69** · `fbrrt_td_lambda` (old code) **−10.71**.
