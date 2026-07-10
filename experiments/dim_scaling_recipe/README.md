# dim_scaling_recipe — the on-policy recipe vs off-policy, across dimension

Full dimension-scaling grid for the recipe that put on-policy SMC methods
above off-policy at d=512 in the `../dim_scaling_consth` probes:
**backward-noising expansion + n_steps=60** (see `run_recipe_cell.py`
docstring for the math and provenance).

## Design

- Identical methodology to `../dim_scaling_consth`: constant 6-nat headroom
  family, coordinate-nested paired instances (same seed ⇒ same problem across
  every dimension and method), law-fitted hyperparameters, 15k steps,
  plateau = tail mean of smoothed validation reward, frac_closed =
  (plateau − E_base)/6.
- Arms (3): `single_seed_mc` × {`expand_ns60`, `expand_ns60_sub`},
  `single_seed_td_lambda` × `expand_ns60`. The `_sub` variant subsamples each
  epoch back to the method's law cadence (freshness-matched control; best ssmc
  variant at d=512). ssmc-td gets only the full variant — its TD(λ)
  bootstrapping *needs* the slower cadence (probe 3: +6.5 full vs +1.8 sub).
- Dims {2, 8, 16, 32, 64, 128, 256, 512} × 30 nested seeds per cell.
- Controls are NOT re-run: `../dim_scaling_consth/results/grid_*.json`
  (off_policy, ssmc, ssmc-td law configs) used the identical protocol on the
  same instances/seeds and are read directly by `plot_recipe.py`.
- d=512 seeds 0–9 for all three arms are carried over from the probe-3 runs
  (identical code path, per-seed deterministic seeding); the cell runner
  resumes those cells at seed 10.

## Frozen code

`problem_consth.py`, `sweep_consth.py`, `fit_consth.py`,
`fitted_models_consth.json`, `hparam_transforms_consth.py` are verbatim copies
from `../dim_scaling_consth` (checkpointed at the git commit recorded in this
directory's first commit); `run_recipe_cell.py` inlines the expansion /
subsample functions from `probe3_cell.py`. Local copies shadow the originals
via sys.path order. Shared base modules still come from
`../dim_scaling_bs4` (committed, stable).

## Run

    bash master_recipe.sh          # POOL=5, big dims first, resumable
    python plot_recipe.py          # writes recipe_vs_offpolicy.png + summary.md

Results land in `results/recipe_<arm>_<method>_d<dim>.json`; per-cell logs in
`logs/`. See `REPORT.md` (written after the grid completes) for the analysis.
