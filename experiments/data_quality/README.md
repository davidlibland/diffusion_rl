# Data-quality experiments

Bias/variance analysis of the regression targets produced by the on-policy
sampling methods, compared against the analytical value function `V_anal(x,t)`.
Runs are kept in date-marked folders (marked by the date the run was produced):

| folder | what | code |
|---|---|---|
| [`2026-05-28/`](2026-05-28/) | Original full 19-stage analysis (`data_quality_v2.py`) covering all sampling methods. **Stale for FBRRT and the τ-methods** — see its `NOTE_methods_changed.md`. | pre-fix (`master`/`8c5782e`) |
| [`2026-06-10/`](2026-06-10/) | **FBRRT re-run** after the FBRRT FBSDE backward-pass fixes (B, D, A/C). Controlled OLD-vs-NEW comparison of the four FBRRT estimators. See [`2026-06-10/REPORT.md`](2026-06-10/REPORT.md). | branch `fix-fbrrt-fbsde-targets` |

The 2026-06-10 run focuses on the FBRRT methods (the ones whose code changed); the
non-FBRRT stages in the 2026-05-28 run are unaffected by the FBRRT fixes and were
not re-run here.
