# dim_scaling_lawv2 — recipe knobs inside the hyperparameter laws

The `../dim_scaling_recipe` grid showed the fixed recipe (expansion +
n_steps=60) inverts the dimension trend: −14 points vs the law configs at
d=8, +10 at d=512. This study makes the recipe knobs part of the
per-dimension hyperparameter laws so a single config *family* can recover
the envelope:

- `n_steps` — re-swept on a log scale 12..120 (old law: constant 19, an
  artifact of the d≤128 anchor blind spot),
- `expand_frac` ∈ [0,1] — fraction of rows given one backward-noising
  expansion sample (0 = old law config, 1 = probe-3 recipe),
- `epoch_rows` ∈ [768, 16384] log — cap on rows per generated epoch
  (uniform subsample after expansion), the dataset-regeneration-cadence
  knob that the probe-3 `*_sub` arms showed matters.

Methods: `single_seed_mc`, `single_seed_td_lambda` (the two that reach/beat
off-policy). **Anchors now include d=512.** Everything else is the frozen
consth methodology (constant 6-nat headroom, nested paired instances, TPE +
Hyperband anchors at 5k steps, LCB trial selection, Theil-Sen laws on log(d)
with 10% LOO bias toward constants, grid at 15k steps × 30 seeds).

## Pipeline

    bash master_lawv2.sh     # A: anchors (2×4 cells, 60 trials each)
                             # B: fit_lawv2.py -> fitted_models_lawv2.json
                             # C: grid (2×8 cells × 30 seeds)
                             # D: plot_lawv2.py -> lawv2_vs_offpolicy.png

Controls are reused, not re-run: off-policy + law-v1 curves from
`../dim_scaling_consth/results`, fixed-recipe curves from
`../dim_scaling_recipe/results` (identical protocol, same instances/seeds).

## Files

- `hparam_transforms_lawv2.py` — search-space spec + per-method layout.
- `sweep_lawv2.py` — anchor sweep (search space incl. recipe knobs;
  `build()` = frozen consth build + `make_augment`).
- `fit_lawv2.py` — law fitting / `hparams_for_dim` (4 anchor dims).
- `run_lawv2_cell.py` — one grid cell, 30 nested seeds.
- `plot_lawv2.py`, `master_lawv2.sh`, `REPORT.md` (written when complete).
- Frozen copies: `problem_consth.py`, `sweep_consth.py` (checkpointed at the
  git commit that adds this directory).
