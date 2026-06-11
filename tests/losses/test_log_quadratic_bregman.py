"""Numerical-stability tests for the log-quadratic Bregman divergence.

The loss VALUE is only used to monitor training; what must be stable and
correct is the GRADIENT w.r.t. the prediction.  The implementation returns the
exact closed form  dL/dp = p*(e^p - e^q)/(e^p - 1)  from a custom backward,
evaluated branch-wise so it is finite for ALL float inputs.  The naive
autograd of the raw formula is NaN for p >~ 88 and -inf for q >~ 88 in
float32 -- one prediction spike then poisons the weights (this killed FBRRT
convergence runs in the bs4_moons 2026-06-10 experiment).
"""

import math

import pytest
import torch

from diffusion_rl.algorithms.spence import Spence1mExp
from diffusion_rl.losses.log_quadratic_bregman import (
    _log_quadratic_bregman_value,
    log_quadratic_bregman_divergence,
    log_quadratic_bregman_grad,
)


def _reference_loss(p, q):
    """The original (unstabilised) implementation, via the Spence Function."""
    quadratic_part = 0.5 * q.exp() * (p**2 - q**2)
    correction = torch.expm1(q) * (Spence1mExp.apply(p) - Spence1mExp.apply(q))
    return quadratic_part + correction


def _closed_form_grad_f64(p, q):
    """dL/dp = p*(e^p - e^q)/(e^p - 1), naively in float64 (healthy regime)."""
    return p * (p.exp() - q.exp()) / torch.expm1(p)


# ---------------------------------------------------------------------------
# healthy regime: value and gradient must match the original implementation
# ---------------------------------------------------------------------------
def test_value_matches_reference_in_healthy_regime():
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        p = torch.linspace(-39, 8, 300, dtype=dtype)
        q = torch.linspace(-35, 2, 300, dtype=dtype).flip(0)
        new = log_quadratic_bregman_divergence(p, q)
        ref = _log_quadratic_bregman_value(p, q)
        assert torch.allclose(new, ref, rtol=1e-6, atol=1e-7), dtype


def test_gradient_matches_autograd_of_reference():
    p = torch.linspace(-39, 8, 300, dtype=torch.float64, requires_grad=True)
    q = torch.linspace(-35, 2, 300, dtype=torch.float64).flip(0)
    (g_ref,) = torch.autograd.grad(_reference_loss(p, q).sum(), p)
    p2 = p.detach().clone().requires_grad_(True)
    (g_new,) = torch.autograd.grad(
        log_quadratic_bregman_divergence(p2, q).sum(), p2)
    assert torch.allclose(g_new, g_ref, rtol=1e-10, atol=1e-12)
    # and against the closed form directly
    g_cf = _closed_form_grad_f64(p.detach(), q)
    assert torch.allclose(g_new, g_cf, rtol=1e-10, atol=1e-12)


def test_gradient_zero_at_minimum_and_limit_at_p_zero():
    q = torch.tensor([-7.0, -1.0, 0.0, 1.0], dtype=torch.float64)
    # Bregman divergence is minimised at p == q: gradient must vanish there.
    g = log_quadratic_bregman_grad(q.clone(), q)
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-14)
    # p == 0 is a removable singularity with limit -expm1(q).
    g0 = log_quadratic_bregman_grad(torch.zeros_like(q), q)
    assert torch.allclose(g0, -torch.expm1(q), rtol=1e-12)
    # continuity across p = 0 (the gradient itself moves O(eps) per step,
    # so the tolerance must admit that)
    eps = 1e-7
    gm = log_quadratic_bregman_grad(torch.full_like(q, -eps), q)
    gp = log_quadratic_bregman_grad(torch.full_like(q, +eps), q)
    assert torch.allclose(gm, g0, rtol=1e-5, atol=2 * eps)
    assert torch.allclose(gp, g0, rtol=1e-5, atol=2 * eps)


# ---------------------------------------------------------------------------
# pathological regime: everything stays finite, gradient stays correct
# ---------------------------------------------------------------------------
def test_old_implementation_was_unstable():
    """Documents the failure this fix removes (float32)."""
    p = torch.tensor([89.0], requires_grad=True)
    q = torch.tensor([-5.0])
    (g,) = torch.autograd.grad(_reference_loss(p, q).sum(), p)
    # The ORIGINAL loss expression with naive autograd through exp/expm1
    # produced NaN here; Spence1mExp.backward has since been stabilised, so
    # the reference path may now be finite -- the loss VALUE, however, still
    # overflows for large q:
    lv = _reference_loss(torch.tensor([-5.0]), torch.tensor([100.0]))
    assert not torch.isfinite(lv).all()
    del g


def test_extreme_inputs_finite_loss_and_gradient_float32():
    extremes_p = [-1e4, -616.0, -89.0, -5.0, 0.0, 5.0, 89.0, 200.0]
    extremes_q = [-1e4, -616.0, -89.0, -5.0, 0.0, 100.0, 200.0]
    P, Q = torch.meshgrid(
        torch.tensor(extremes_p), torch.tensor(extremes_q), indexing="ij")
    p = P.reshape(-1).clone().requires_grad_(True)
    q = Q.reshape(-1).clone()
    loss = log_quadratic_bregman_divergence(p, q)
    assert torch.isfinite(loss).all(), "loss value not finite at extremes"
    (g,) = torch.autograd.grad(loss.sum(), p)
    assert torch.isfinite(g).all(), "gradient not finite at extremes"


def test_extreme_gradients_are_correct():
    """Spot-check the stable gradient against float64 closed-form limits."""
    f64 = torch.float64

    def g(pv, qv):
        return log_quadratic_bregman_grad(
            torch.tensor([pv], dtype=f64), torch.tensor([qv], dtype=f64)
        ).item()

    # p large positive: dL/dp -> p * (1 - e^{q-p}) / (1 - e^{-p}) -> p
    assert g(200.0, -5.0) == pytest.approx(200.0, rel=1e-12)
    # p very negative, q moderate: dL/dp -> p * (0 - e^q) / (0 - 1) = p e^q
    assert g(-616.0, -5.0) == pytest.approx(-616.0 * math.exp(-5.0), rel=1e-9)
    # q very negative: dL/dp -> p e^p / (e^p - 1)
    pv = -5.0
    expect = pv * math.exp(pv) / math.expm1(pv)
    assert g(pv, -1e4) == pytest.approx(expect, rel=1e-12)
    # huge q (e^q overflows float32): finite, correct in float64
    pv, qv = -5.0, 100.0
    expect = pv * (math.exp(pv) - math.exp(qv)) / math.expm1(pv)
    assert g(pv, qv) == pytest.approx(expect, rel=1e-12)


def test_spence_backward_finite_everywhere():
    x = torch.tensor([-700.0, -89.0, -1.0, 0.0, 1.0, 89.0, 700.0],
                     requires_grad=True)
    (g,) = torch.autograd.grad(Spence1mExp.apply(x).sum(), x)
    assert torch.isfinite(g).all()
    # d/dx Li2(1 - e^x) = x / expm1(-x):  -> -x for large x, -> 0 for x -> -inf
    assert g[-1].item() == pytest.approx(-700.0, rel=1e-6)
    assert g[0].item() == pytest.approx(0.0, abs=1e-12)


def test_target_gradient_raises():
    p = torch.randn(4)
    q = torch.randn(4, requires_grad=True)
    loss = log_quadratic_bregman_divergence(p, q).sum()
    with pytest.raises(NotImplementedError):
        loss.backward()
