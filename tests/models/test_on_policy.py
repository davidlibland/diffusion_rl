"""Tests for on-policy SMC algorithms in diffusion_rl.models.on_policy."""

import math

import pytest
import torch

from diffusion_rl.models.on_policy import (
    OnPolicySMCDataset,
    _girsanov_log_rho,
    _sde_step,
    ancestral_mc_td_lambda,
    ancestral_td_lambda,
    grad_value_guidance,
    one_step_bootstrap,
    single_seed_mc,
    single_seed_td_lambda,
)


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


# ---------------------------------------------------------------------------
# Guided proposals (drift + scale * 2a * grad V with Girsanov-corrected weights)
# ---------------------------------------------------------------------------
#
# Oracle setup used throughout: zero base drift, reward r(x) = c*x, so under
# the base process X_1 | X_t = x ~ N(x, 2a(1-t)) and the value function is
# exact in closed form:
#
#     V(x, t) = log E[exp(c X_1) | X_t = x] = c*x + c^2 * a * (1 - t),
#     grad_x V = c   (constant).
#
# With zero base drift and a constant guidance control the Euler-Maruyama
# chain is EXACT (no discretization bias), so exp(target - V_oracle) must
# have mean 1 up to Monte Carlo noise for every sampler, guided or not.

_C = 0.5
_A = 1.0


def _oracle_problem():
    def drift(x, t):
        return torch.zeros_like(x)

    def value(x, t):
        return _C * x.squeeze(-1) + _C**2 * _A * (1.0 - t.squeeze(-1))

    def h(x):
        return _C * x.squeeze(-1)

    return drift, value, h


def _oracle_v(x, t):
    """Exact V on flat outputs: x (M, 1), t (M,) -> (M,)."""
    return _C * x.squeeze(-1) + _C**2 * _A * (1.0 - t)


def test_girsanov_log_rho_matches_gaussian_density_ratio():
    """log_rho must equal the exact log-ratio of the two Euler-Maruyama kernels.

    x' = x + (f + 2a u) dt + db with db ~ N(0, 2a dt I): the base kernel is
    N(x + f dt, 2a dt I), the guided kernel N(x + (f + 2a u) dt, 2a dt I).
    """
    torch.manual_seed(0)
    n, d, a, dt = 64, 3, 0.7, 0.05
    x = torch.randn(n, d)
    f = torch.randn(n, d)
    u = torch.randn(n, d)
    db = math.sqrt(2 * a * dt) * torch.randn(n, d)
    x_next = x + (f + 2 * a * u) * dt + db

    scale = math.sqrt(2 * a * dt)
    log_p_base = (
        torch.distributions.Normal(x + f * dt, scale).log_prob(x_next).sum(-1)
    )
    log_p_guided = (
        torch.distributions.Normal(x + (f + 2 * a * u) * dt, scale)
        .log_prob(x_next)
        .sum(-1)
    )

    log_rho = _girsanov_log_rho(u, db, a, dt)
    assert torch.allclose(log_rho, log_p_base - log_p_guided, atol=1e-5)


def test_sde_step_applies_guidance_drift():
    """Guided step must shift the proposal by exactly 2a*u*dt vs the unguided one."""
    n, d, a, dt, t = 32, 2, 0.4, 0.1, 0.3
    x = torch.randn(n, d)

    def drift(x_, t_):
        return -x_

    u0 = 0.8

    torch.manual_seed(7)
    x1, lr1 = _sde_step(x, drift, a, t, dt, n, 1, d, torch.device("cpu"))
    torch.manual_seed(7)
    x2, lr2 = _sde_step(
        x, drift, a, t, dt, n, 1, d, torch.device("cpu"),
        guidance=lambda x_, t_: torch.full_like(x_, u0),
    )

    assert torch.allclose(x2 - x1, torch.full_like(x, 2 * a * u0 * dt), atol=1e-6)
    assert torch.all(lr1 == 0)
    db = x1 - x - drift(x, None) * dt  # realised noise (same seed)
    u = torch.full_like(x, u0)
    expected = -(u * db).sum(-1) - a * (u * u).sum(-1) * dt
    assert torch.allclose(lr2, expected, atol=1e-5)


def test_grad_value_guidance_returns_scaled_gradient():
    def value(x, t):
        return (x**2).sum(dim=-1) + t.squeeze(-1)

    guidance = grad_value_guidance(value, scale=0.5)
    x = torch.randn(16, 3)
    t = torch.rand(16, 1)
    with torch.no_grad():  # must work inside no_grad, like the samplers
        u = guidance(x, t)
    assert torch.allclose(u, 0.5 * 2 * x, atol=1e-5)
    assert not u.requires_grad

    # Frozen-parameter module (the EMA-shadow regime used as smc_value):
    # the input gradient must still flow.
    lin = torch.nn.Linear(3, 1)
    lin.requires_grad_(False)
    guidance_ema = grad_value_guidance(lambda x_, t_: lin(x_).squeeze(-1), 1.0)
    with torch.no_grad():
        u_ema = guidance_ema(x, t)
    assert torch.allclose(u_ema, lin.weight.expand(16, 3), atol=1e-6)


def _run_sampler(method, guidance, log_tau, seed, lambda_eff=0.5):
    """Run one sampler on the oracle problem; return (all_x, all_t, all_tgt)."""
    drift, value, h = _oracle_problem()
    device = torch.device("cpu")
    common = dict(
        drift=drift, value=value, log_tau=log_tau, h=h, a=_A,
        dim=1, n_steps=4, device=device,
    )
    torch.manual_seed(seed)
    if method == "one_step_bootstrap":
        # one_step_bootstrap consumes (B, N, d)-shaped callables (the dataset
        # wraps them via drift_fn/value_fn/...); adapt the flat oracle fns.
        def to3d(fn):
            def fn3(x, t):
                B, N, d = x.shape
                return fn(x.reshape(B * N, d), t).reshape(B, N, 1)

            return fn3

        def h3(x):
            B, N, d = x.shape
            return h(x.reshape(B * N, d)).reshape(B, N, 1)

        guidance_3d = (
            None
            if guidance is None
            else lambda x, t: guidance(x.reshape(-1, x.shape[-1]), t).reshape(
                x.shape
            )
        )
        common.update(
            drift=lambda x, t: torch.zeros_like(x),
            value=to3d(value),
            log_tau=to3d(log_tau),
            h=h3,
        )
        return one_step_bootstrap(
            batch_size=256, mc_samples=32, guidance=guidance_3d, **common
        )
    if method == "ancestral_td_lambda":
        return ancestral_td_lambda(
            lambda_eff=lambda_eff, batch_size=1024, mc_samples=8,
            guidance=guidance, **common,
        )
    if method == "ancestral_mc_td_lambda":
        return ancestral_mc_td_lambda(
            lambda_eff=lambda_eff, batch_size=1024, mc_samples=8,
            guidance=guidance, **common,
        )
    if method == "single_seed_td_lambda":
        return single_seed_td_lambda(
            lambda_eff=lambda_eff, batch_size=1024, mc_samples=16,
            guidance=guidance, **common,
        )
    if method == "single_seed_mc":
        return single_seed_mc(
            batch_size=1024, mc_samples=16, guidance=guidance, **common
        )
    raise ValueError(method)


_ALL_METHODS = [
    "one_step_bootstrap",
    "ancestral_td_lambda",
    "ancestral_mc_td_lambda",
    "single_seed_td_lambda",
    "single_seed_mc",
]


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_guided_zero_control_is_bitwise_identical_to_unguided(method):
    """guidance == 0 must reproduce the unguided sampler exactly (same RNG)."""
    _, value, _ = _oracle_problem()

    def zero_guidance(x, t):
        return torch.zeros_like(x)

    out_a = _run_sampler(method, None, value, seed=123)
    out_b = _run_sampler(method, zero_guidance, value, seed=123)
    for ta, tb in zip(out_a, out_b):
        assert torch.equal(ta, tb)


@pytest.mark.parametrize("method", _ALL_METHODS)
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_guided_targets_unbiased_against_oracle(method, scale):
    """exp(target - V_oracle) must average to 1 under the guided proposal.

    The guidance here is u = scale * grad V = scale * c; the Girsanov
    correction must compensate it exactly, leaving the targets unbiased for
    the BASE-process value function.

    scale=1.0 is special: with the exact optimal control and exact V the
    Girsanov exponent cancels the value fluctuation pointwise (the
    zero-variance importance-sampling identity), so the per-sample ratio is
    exactly 1 -- a sharp algebraic check of the correction's sign and
    factors.  scale=0.5 (undershoot) and scale=2.0 (overshoot) leave real
    variance and exercise the statistical unbiasedness.
    """
    _, value, _ = _oracle_problem()

    def guidance(x, t):
        return torch.full_like(x, scale * _C)

    all_x, all_t, all_tgt = _run_sampler(method, guidance, value, seed=0)
    ratio = torch.exp(all_tgt - _oracle_v(all_x, all_t))
    m = ratio.mean().item()
    assert abs(m - 1.0) < 0.06, f"{method} guided (scale={scale}) biased: {m}"


@pytest.mark.parametrize(
    "method", ["ancestral_td_lambda", "ancestral_mc_td_lambda"]
)
def test_guided_targets_twist_independent(method):
    """With guidance on, the targets must still not depend on the twist tau.

    The twist only steers resampling; together with the Girsanov-corrected
    weights it must cancel out of the targets (FK-steering consistency).

    Uses a non-optimal scale (0.5): at scale=1 with exact V the targets are
    pointwise exact (see above) and the twist never gets a chance to leak.
    """
    _, value, _ = _oracle_problem()

    def tau_unrelated(x, t):
        return 0.4 * x.squeeze(-1) ** 2 - 0.2 * t.squeeze(-1)

    def guidance(x, t):
        return torch.full_like(x, 0.5 * _C)

    means = {}
    for name, tau in [("value", value), ("unrelated", tau_unrelated)]:
        all_x, all_t, all_tgt = _run_sampler(method, guidance, tau, seed=3)
        ratio = torch.exp(all_tgt - _oracle_v(all_x, all_t))
        means[name] = ratio.mean().item()
        assert abs(means[name] - 1.0) < 0.08, (
            f"{method} guided, tau={name}: biased ratio {means[name]}"
        )
    assert abs(means["value"] - means["unrelated"]) < 0.08


def test_guided_terminal_samples_match_tilted_distribution():
    """Guided sampling must leave the SMC terminal law unchanged.

    With tau = exact V, the SMC sweep targets the tilted terminal law
    p(x) exp(c x) / Z = N(2ac, 2a) at t=1.  Guidance changes the proposals
    but the corrected weights must keep the terminal samples distributed
    identically, so the t=1 sample mean must match 2ac with and without
    guidance.

    Self-normalised resampling has an O(1/mc_samples) bias in this mean
    (resampling happens within groups of mc_samples particles), so use a
    large group size to isolate the guidance effect from that baseline bias.
    """
    drift, value, h = _oracle_problem()

    def guidance(x, t):
        return torch.full_like(x, _C)

    target_mean = 2 * _A * _C
    for g in (None, guidance):
        torch.manual_seed(11)
        all_x, all_t, _ = ancestral_td_lambda(
            drift=drift, value=value, log_tau=value, h=h, a=_A,
            lambda_eff=0.5, batch_size=64, mc_samples=128, dim=1,
            n_steps=4, device=torch.device("cpu"), guidance=g,
        )
        x1 = all_x[all_t == 1.0]
        m = x1.mean().item()
        # std of the tilted law is sqrt(2a); ~8k samples -> stderr ~ 0.016;
        # remaining self-normalisation bias at N=128 is ~1-2%.
        assert abs(m - target_mean) < 0.1, (
            f"terminal mean {m} != {target_mean} (guided={g is not None})"
        )


@pytest.mark.parametrize("method", ["single_seed_mc", "one_step_bootstrap"])
def test_dataset_guidance_scale_smoke(method):
    """OnPolicySMCDataset with smc_guidance_scale > 0 yields finite batches.

    Exercises grad_value_guidance end-to-end (autograd through smc_value
    inside the dataset's torch.no_grad() generation loop), including the 3D
    guidance_fn wrapper used by one_step_bootstrap.
    """
    drift, value, h = _oracle_problem()

    # smc_value must be differentiable w.r.t. x -- the oracle V is.
    ds = OnPolicySMCDataset(
        dim=1,
        drift=drift,
        value=value,
        smc_value=value,
        reward=h,
        device=torch.device("cpu"),
        sampling_method=method,
        a=_A,
        batch_size=64,
        n_steps=4,
        mc_samples_per_step=4,
        smc_guidance_scale=0.7,
    )
    it = iter(ds)
    for _ in range(8):
        y, x, t, w = next(it)
        assert torch.isfinite(x).all()
        assert torch.isfinite(y).all()
        assert torch.isfinite(t).all()
        assert (w == 1.0).all()  # non-FBRRT samplers keep uniform weights
