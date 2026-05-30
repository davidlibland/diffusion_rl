# experiments/misc

Archived exploratory experiments from the old `notebooks/` directory, grouped by
family and prefixed with their (git-history) authoring date so they sort
chronologically. **These may be outdated** — they predate later fixes to the
methods in `src/diffusion_rl/models/`. Treat results here as historical unless
re-run against current code.

Shared inputs (notably `analytical_target.json`) and the generic plotter live in
[`../common/`](../common/). Scripts use repo-root-relative paths — run them from
the repo root.

| directory | what it explores |
|---|---|
| `2026-03-24_exploratory_notebooks` | Scratch notebooks: `sandbox`, `grpo`, `diffusion_llm`. |
| `2026-03-24_moons_baseline` | Initial moons off/on-policy notebooks + base sweeps. |
| `2026-03-24_training_runs` | Plain convergence / long training runs. |
| `2026-03-24_lr_sweeps` | Learning-rate sweeps (on- and off-policy). |
| `2026-03-24_oracle_onpolicy` | On-policy runs with the oracle (analytical) value. |
| `2026-03-25_warmstart` | Warm-starting experiments + frozen-SMC variant. |
| `2026-03-25_one_step_bootstrap` | One-step bootstrap target experiment. |
| `2026-03-25_lambda_logz_sweep` | λ sweep with logZ decomposition. |
| `2026-03-25_ancestral_training_sweep` | Ancestral TD(λ) training sweep. |
| `2026-03-25_eval_value_functions` | Evaluation of trained value functions. |
| `2026-03-25_targeted_runs` | Targeted single-config runs. |
| `2026-05-06_fbrrt_sweeps` | FBRRT family: alpha/grad-decay/λ/titrate/mixed sweeps. |
| `2026-05-06_batch_size_sweep` | Batch-size sweep (bs1–bs256) + plots/report. |
| `2026-05-06_dimension_scaling` | Dimension-scaling experiment (provenance for `../dim_scaling_bs4/`). |
| `2026-05-06_ema_experiment` | EMA-of-value experiment. |
| `2026-05-06_lambda_training_sweep` | λ training sweep. |
| `2026-05-06_mc_samples_sweep` | MC-samples sweep (+ SSMC variant) + report. |
| `2026-05-06_mixed_training` | Mixed on/off-policy training variants. |
| `2026-05-06_alternating_training` | Alternating on/off-policy training. |
| `2026-05-06_warmstart_final` | Final warm-start experiment. |
| `2026-05-06_run_single_frac` | Single off-policy-fraction run helper. |
| `2026-05-28_ssmc_sweeps` | SSMC family: k / k·t / k·V / mc-bias / shuffle / seeds / vs-offpolicy. |
| `2026-05-28_bs_sweep` | Second batch-size sweep (+ balanced variant). |
| `2026-05-28_offpolicy_seeds` | Off-policy multi-seed baseline (its `offpolicy_seeds_results.json` is read by several ssmc scripts). |
| `2026-05-28_onpolicy_frac1_seeds` | On-policy (frac=1) multi-seed runs. |
| `2026-05-29_dim_scaling_methods_check` | Method-validation check (provenance for `../dim_scaling_bs4/`). |
