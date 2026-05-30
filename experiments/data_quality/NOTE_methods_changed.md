# ⚠️ Stale results — methods changed, re-run required

The analyses in this directory (`data_quality.py`, `data_quality_v2.py`, and the
`dq2_*` / `data_quality_*` plots and JSON results) were produced **before**
recent changes to the on-policy method implementations in
[`src/diffusion_rl/models/on_policy.py`](../../src/diffusion_rl/models/on_policy.py).

The following methods were modified and **must be re-run** before these results
should be trusted:

- **`ancestral_mc_td_lambda`** — the multi-step / duplicate-averaging backward
  pass was corrected (the SMC twist used to leak into the regression targets,
  producing an O(1), twist-dependent bias for `lambda_eff > 0`). See the
  regression test
  `tests/models/test_on_policy.py::test_ancestral_mc_td_lambda_target_is_tau_independent_and_unbiased`.
- **`single_seed_mc` / `single_seed_td_lambda` (SSMC / SSMC-TD(λ))** — also
  modified.

Any bias/variance numbers, plots, or conclusions in this directory that involve
these methods reflect the **old** implementations. Regenerate the affected
stages (e.g. `python experiments/data_quality/data_quality_v2.py`, then
`python experiments/data_quality/dq2_clean_plots.py`) against the current code
before citing them.
