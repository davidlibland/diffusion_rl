# bs4_moons — on-policy vs off-policy at gradient batch size 4

Benchmark comparing all on-policy sampling methods and the off-policy baseline
on the moons/GMM task at BS=4 (Optuna sweep → 5-seed confirm → 50k-step
convergence; detrended-SEM LCB objective). Runs live in date-marked folders:

| folder | what | code |
|---|---|---|
| [`2026-05-28/`](2026-05-28/) | Original full study (all 10 methods; summary in [`bs4_experiments_summary.md`](2026-05-28/bs4_experiments_summary.md)). Includes the 2026-05-29 `ancestral_mc_td_lambda` re-tune. **⚠️ The FBRRT results predate the FBRRT FBSDE fixes** (commits `57926d6`, `13ef36a`, `c6b70ee`: entropy weights moved off the target, ancestor-aligned TD(λ), corrected CV driver/Malliavin scaling, child-time driver gradients). The FBRRT rows there (`fbrrt_td_lambda` plateau −10.71, "weakest and slowest") reflect the **old, biased estimators** and should not be cited as the methods' performance. Non-FBRRT rows are unaffected. | pre-fix (≤ `3ec4f7b`) |
| [`2026-06-10/`](2026-06-10/) | **FBRRT re-tune** under the fixed estimators (one 80-trial study per method, 9-way parallel), same task/objective/budgets. See [`2026-06-10/REPORT.md`](2026-06-10/REPORT.md). **Outcome:** `fbrrt_cv` plateau **−6.42** (was −10.71 with old code) — beats off-policy (−9.69) and `single_seed_mc` (−6.78); the SMC-twist TD(λ) methods (−5.16/−5.67) still lead. | post-fix (`2fd7535`+) |

Plateau reward at 50k, BS=4 (multi-seed where available; see the
[2026-06-11 addendum](2026-06-10/REPORT.md) for the stabilization work):
`single_seed_td_lambda` **−5.16** · `ancestral_mc_td_lambda` (re-tuned) **−5.67**
· **`fbrrt_cv` (fixed, tuned policy-EMA) −5.58 ± 0.67 (1/3 late deaths)**
· **`fbrrt` + EMA(0.99) target network −6.34 ± 0.56 (0/3 deaths)**
· `single_seed_mc` **−6.78** · off-policy **−9.69** · `fbrrt_td_lambda` (old code) **−10.71**.
