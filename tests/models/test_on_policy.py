"""Tests for on-policy SMC algorithms in diffusion_rl.models.on_policy."""

import math

import torch

from diffusion_rl.models.on_policy import ancestral_mc_td_lambda, single_seed_mc


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


def test_ancestral_mc_td_lambda_target_is_tau_independent_and_unbiased():
    r"""`ancestral_mc_td_lambda` targets must be unbiased when the value is exact.

    If exp(value) equals the true value H(x,t) = E[exp(reward(X_T)) | X_t = x],
    then exp(tgt) must be an unbiased estimate of H(x,t).  Crucially this must
    hold *independently of the twist* `log_tau`, which only affects resampling
    and must cancel out of the targets.

    We use an analytically solvable case: zero drift, a=1, so under the sampled
    SDE X_1 | X_t = x ~ N(x, 2(1-t)).  With reward h(x)=c*x the value is exact in
    closed form, H(x,t) = exp(c*x + c^2*(1-t)).  We check the t=0 generation,
    where every particle sits at the origin, so E[exp(tgt) | X_0=0] is just the
    mean of exp(tgt) and must equal H(0,0) = exp(c^2).

    This is a regression test for a former bug where the multi-step (lambda>0)
    term mis-indexed the resampling weights, leaking the twist into the target:
    the buggy code returned ~-8% bias for one twist and ~+16% for another (and a
    product-of-means variant diverged to +60%..+600%).  The fix averages the
    per-child product w(child)*rho_hat(child) over a parent's resampled copies,
    so the twist cancels and only a small O(dt) smoothing residual remains.
    """
    c = 0.5
    a = 1.0
    h00 = math.exp(c**2)  # true H(0,0) = E[exp(c * X_1) | X_0 = 0]
    device = torch.device("cpu")

    def drift(x, t):
        return torch.zeros_like(x)

    def value(x, t):
        # Exact log-value: log H(x,t) = c*x + c^2*(1-t).
        return c * x.squeeze(-1) + c**2 * (1.0 - t.squeeze(-1))

    def h(x):
        # Terminal log-value log H(x,1) = reward = c*x.
        return c * x.squeeze(-1)

    # Two structurally different twists; targets must be (nearly) identical.
    def tau_equals_value(x, t):
        return c * x.squeeze(-1) + c**2 * (1.0 - t.squeeze(-1))

    def tau_unrelated(x, t):
        return 0.4 * x.squeeze(-1) ** 2 - 0.2 * t.squeeze(-1)

    def t0_mean(lambda_eff, log_tau, seed):
        torch.manual_seed(seed)
        _, all_t, all_tgt = ancestral_mc_td_lambda(
            drift=drift,
            value=value,
            log_tau=log_tau,
            h=h,
            a=a,
            lambda_eff=lambda_eff,
            batch_size=1024,
            mc_samples=8,
            dim=1,
            n_steps=4,
            device=device,
        )
        return torch.exp(all_tgt[all_t == 0.0]).mean().item()

    # lambda=0 (pure one-step bootstrap): exactly unbiased and twist-independent.
    for log_tau in (tau_equals_value, tau_unrelated):
        m = t0_mean(0.0, log_tau, seed=0)
        assert abs(m - h00) / h00 < 0.04, f"lambda=0 biased: {m} vs {h00}"

    # lambda=1 (pure multi-step): the twist must cancel.  The remaining O(dt)
    # smoothing bias is ~1-2% at n_steps=4, far below the buggy >=8% deviations.
    m_val = t0_mean(1.0, tau_equals_value, seed=1)
    m_unr = t0_mean(1.0, tau_unrelated, seed=1)
    assert abs(m_val - h00) / h00 < 0.06, f"lambda=1 biased (tau=value): {m_val}"
    assert abs(m_unr - h00) / h00 < 0.06, f"lambda=1 biased (tau=unrelated): {m_unr}"
    # Twist-independence is the core regression assertion (buggy gap was ~24%).
    assert abs(m_val - m_unr) / h00 < 0.04, (
        f"target depends on the twist: {m_val} vs {m_unr}"
    )
