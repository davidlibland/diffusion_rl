# experiments/common

Shared assets and standalone utilities reused across the experiments in
[`../misc/`](../misc/) (and the curated experiments in `../bs4_moons/`,
`../data_quality/`). These are **not** an importable package — the scripts in
this repo are self-contained and duplicate their own setup; what is genuinely
shared lives here as data + standalone scripts.

| file | role |
|---|---|
| `analytical_target.py` → `analytical_target.json` | Computes/stores the analytical target constants (e.g. `E_opt`) for the moons toy problem. **The JSON is read by ~50 experiment scripts** as `experiments/common/analytical_target.json`. |
| `compute_analytical_value.py` | Computes the analytical value function; produces `analytical_value_heatmaps.png`, `analytical_vs_network.png`. |
| `analytical_value_function.md` | Write-up of the analytical value function (references the two PNGs above). |
| `sanity_check_analytical_drift.py` | Sanity-checks the analytical drift against the SDE. |
| `plot_sweep_results.py` | Generic plotter that reads Lightning CSV logs and produces the comparison figures `plot_sampling_methods.png`, `plot_loss_type.png`, `plot_off_vs_on.png`, `plot_lambda_sweep.png`. |

Paths inside these scripts are repo-root-relative — run them from the repo root.
