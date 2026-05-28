"""Tests for on-policy SMC algorithms in diffusion_rl.models.on_policy."""

import torch

from diffusion_rl.models.on_policy import single_seed_mc


def test_single_seed_mc_uses_h_as_terminal_value():
    """`single_seed_mc` should use `h` (reward) as the terminal value, never `v`.

    We force `value` to return NaN; if any output depended on `value`, it
    would propagate NaN. We also count calls to `h` to confirm it is invoked.
    """
    torch.manual_seed(0)

    batch_size = 3
    mc_samples = 4
    dim = 2
    n_steps = 5
    a = 0.1
    device = torch.device("cpu")
    dtype = torch.float32

    h_call_count = {"n": 0}

    def drift(x, t):
        return torch.zeros_like(x)

    def value(x, t):
        # If `single_seed_mc` ever depends on `value`, NaN will leak into outputs.
        return torch.full(
            (x.shape[0], 1), float("nan"), dtype=x.dtype, device=x.device
        )

    def log_tau(x, t):
        # Smooth, finite log-density-ratio surrogate.
        return -0.5 * (x * x).sum(dim=-1, keepdim=True)

    def h(x):
        h_call_count["n"] += 1
        return -0.25 * (x * x).sum(dim=-1, keepdim=True)

    all_x, all_t, all_tgt = single_seed_mc(
        drift=drift,
        value=value,
        log_tau=log_tau,
        h=h,
        a=a,
        batch_size=batch_size,
        mc_samples=mc_samples,
        dim=dim,
        n_steps=n_steps,
        device=device,
        dtype=dtype,
    )

    # h must have been called at least once (terminal step + exact terminal target).
    assert h_call_count["n"] >= 1, "single_seed_mc never invoked h"

    # Shapes are as documented: n_steps + 1 samples per batch element
    # (one per t_grid point: t=0, t_1, ..., t_{n_steps-1}, t=1).
    assert all_x.shape == (batch_size * (n_steps + 1), dim)
    assert all_t.shape == (batch_size * (n_steps + 1),)
    assert all_tgt.shape == (batch_size * (n_steps + 1),)

    # The t-grid endpoints are included.
    t_unique = torch.unique(all_t)
    assert t_unique.min().item() == 0.0, "t=0 sample missing"
    assert t_unique.max().item() == 1.0, "t=1 sample missing"

    # No NaNs leak through, despite `value` returning NaN.
    assert torch.isfinite(all_x).all(), "all_x contains non-finite values"
    assert torch.isfinite(all_t).all(), "all_t contains non-finite values"
    assert torch.isfinite(all_tgt).all(), "all_tgt contains non-finite values"
