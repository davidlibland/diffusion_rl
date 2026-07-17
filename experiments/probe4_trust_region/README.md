# probe4_trust_region — TRI-TSMC trust-region mechanisms in the training loop

Adapts the trust-region machinery of TRI-TSMC (Wang et al. 2026,
arXiv:2605.25123; PDF in `../../docs/papers/tri_tsmc.pdf`) from its published
regime (inference-time alignment, 3 outer iterations) to our continuous
value-training loop (~49 twist updates per run). We keep our
conditionally-unbiased H-martingale regression targets and use the trust
region purely as **sampling-measure control**. See `probe4_cell.py` docstring
for the mapping (escort path / 1-D dual / chi^2 monotonicity → our twist
schedule, concentration ramp, and weight tempering).

Arms (ssmc, d=512, paired seeds, 15k steps):

| arm | mechanism | replaces | paired control |
|---|---|---|---|
| tr_twist | anchored twist, KL-budgeted lerp per regen (adaptive-decay EMA) | ema_decay + cadence | gridv2 law-v2 (35.3%) |
| tr_oracle_ramp | twist = β·V*, β grows under KL budget | instant oracle twist | probe2 oracle_twist (16.3%), gridv2 (35.3%) |
| tr_unweight | expansion unweighting tempered by the TRI-TSMC dual | 100× weight clamp | probe3 expand_oracle (21.9%) |

For the tr arms epoch_rows is forced to 1216 (fast cadence) so ε — the KL
budget in nats per regeneration — is the sole knob governing measure movement.

Pipeline: `bash run_probe4.sh` — phase 1 scans ε ∈ {0.3, 1.0, 3.0} on seeds
0–2, phase 2 picks the best ε per arm (results/best_eps.json), phase 3
extends the winners to seeds 0–9 (resumes the scan seeds). Diagnostics
(β trajectories, tempering τ) are stored in the per-seed records.

Code dependencies are the frozen `../dim_scaling_lawv2` snapshot (base build,
laws v2) plus `../dim_scaling_consth` law-v1 laws for the tr_unweight control
config. REPORT.md written when complete.
