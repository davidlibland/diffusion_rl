"""Stability tests for the exp-space MSE loss (loss_type="mse")."""

import pytest
import torch

from diffusion_rl.losses.exp_mse import exp_mse, exp_mse_grad


def test_matches_naive_in_healthy_regime():
    torch.manual_seed(0)
    p = torch.linspace(-39, 8, 200, dtype=torch.float64, requires_grad=True)
    q = torch.linspace(-35, 2, 200, dtype=torch.float64).flip(0)
    naive = (p.exp() - q.exp()) ** 2
    (g_ref,) = torch.autograd.grad(naive.sum(), p)
    p2 = p.detach().clone().requires_grad_(True)
    out = exp_mse(p2, q)
    assert torch.allclose(out, naive.detach(), rtol=1e-10)
    (g_new,) = torch.autograd.grad(out.sum(), p2)
    assert torch.allclose(g_new, g_ref, rtol=1e-10)


def test_extremes_finite_where_naive_overflows():
    # naive loss overflows float32 for p > ~44 -- this caused 18% skipped
    # batches in the fbrrt_td_lambda BS=4 convergence run.
    p = torch.tensor([50.0, 100.0, -100.0, -1e4], requires_grad=True)
    q = torch.tensor([-5.0, 200.0, 50.0, -1e4])
    naive = (p.detach().exp() - q.exp()) ** 2
    assert not torch.isfinite(naive).all()
    out = exp_mse(p, q)
    assert torch.isfinite(out).all()
    (g,) = torch.autograd.grad(out.sum(), p)
    assert torch.isfinite(g).all()
    # saturated gradient keeps the correct sign (p=50 >> q=-5: push down)
    assert g[0] > 0


def test_gradient_zero_at_minimum():
    q = torch.tensor([-7.0, -1.0, 0.0, 2.0], dtype=torch.float64)
    g = exp_mse_grad(q.clone(), q)
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-14)


def test_target_gradient_raises():
    p = torch.randn(4)
    q = torch.randn(4, requires_grad=True)
    with pytest.raises(NotImplementedError):
        exp_mse(p, q).sum().backward()
