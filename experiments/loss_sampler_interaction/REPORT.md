# Does the on-policy result depend on which loss is used?

**Question.** S7.2 compares samplers with the loss held at Spence; S5 compares
losses with the sampler held at off-policy. Each is internally valid, but the
quantity one holds fixed is the quantity the other varies — and S5 shows the
fixed choice is suboptimal on S7.2's own metric. Do the two interact?

**Answer: yes, substantially, at d=8.** Two-thirds of the measured on-policy
advantage disappears when the loss is changed.

## Design

`single_seed_mc` vs `off_policy`, at d ∈ {8, 128}, under loss ∈ {quad, logmse},
10 nested paired seeds, 8000 steps, hyperparameters unchanged from the
respective fitted laws. Seeds are nested, so the same problem instance appears
in all four cells and the interaction is a within-instance contrast.

## Results (frac_closed %, paired)

| loss | d | off-policy | on-policy | sampler effect |
|---|---|---|---|---|
| Spence  | 8   | 36.0 | 46.5 | **+10.5** (p<0.001) |
| log-MSE | 8   | 57.5 | 61.1 | **+3.6** (p=0.024) |
| Spence  | 128 | 24.9 | 23.2 | −1.7 (p=0.34) |
| log-MSE | 128 | 33.6 | 32.5 | −1.1 (p=0.37) |

**Difference in differences.** d=8: +6.9, p=0.0048. d=128: −0.5, p=0.82.

**Loss effect on the off-policy arm alone.** log-MSE − Spence: +21.5 at d=8
(p<1e-4), +8.7 at d=128 (p=0.008).

## Readings

1. **The on-policy gain is substantially loss-compensation.** At d=8 it is
   +10.5 under Spence and +3.6 under log-MSE. The sampler is partly making up
   for a baseline handicapped by loss conditioning; improve the conditioning
   and most of the gap closes on its own.
2. **The loss choice dominates the sampler choice on this benchmark.**
   +21.5 (loss, off-policy arm) against +10.5 (sampler, Spence). Reversing the
   usual emphasis: on this task, what to regress with matters more than what to
   sample with.
3. **At d=128 neither matters** — no sampler effect under either loss, no
   interaction. The high-dimensional regime is limited by something both
   changes leave untouched.
4. **The two studies are therefore not separable.** S7.2's headline is
   conditional on its loss and must be reported that way.

## Caveat

log-MSE is a *biased* estimator of V (S5): it wins this metric and loses the
value. So (2) is not advice to adopt it — it is a statement about what
`frac_closed` attributes to sampling versus to conditioning.
