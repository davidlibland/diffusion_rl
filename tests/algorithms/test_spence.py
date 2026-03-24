import numpy as np
import pytest
import torch
from scipy.special import spence as scipy_spence

from diffusion_rl.algorithms.spence import Spence1mExp, spence_1mexp_value


def test_spence_1mexp():
    """Tests the spence_1mexp function. We only need a mild atol of 0.05"""
    x = np.linspace(-10, 20, 100)
    y_ = spence_1mexp_value(torch.from_numpy(x)).numpy()
    y_scipy = scipy_spence(np.exp(x))

    np.testing.assert_allclose(y_, y_scipy, rtol=0, atol=0.05)


def test_Spence1mExp_value():
    """Tests the derivative of the spence function"""
    x = np.linspace(-10, 20, 100)
    y_ = Spence1mExp.apply(torch.from_numpy(x)).numpy()
    y_scipy = scipy_spence(np.exp(x))

    np.testing.assert_allclose(y_, y_scipy, rtol=0, atol=0.05)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_Spence1mExp_dtypes(dtype):
    x = torch.linspace(-10, 20, 100, dtype=dtype)
    y = Spence1mExp.apply(x)

    assert y.dtype == x.dtype


def test_Spence1mExp_derivative():
    """Tests the derivative of the spence function"""
    x = torch.linspace(-10, 20, 1000)
    x_np = x.clone().numpy()
    x.requires_grad_(True)
    y_ = Spence1mExp.apply(x).sum()
    y_.backward()
    y_scipy = scipy_spence(np.exp(x_np))
    # Compute the differences:
    scipy_deriv = np.diff(y_scipy) / np.diff(x_np)

    np.testing.assert_allclose(x.grad.numpy()[:-1], scipy_deriv, rtol=0.01, atol=0.1)
