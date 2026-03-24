import torch

from diffusion_rl.algorithms.spence import Spence1mExp


def log_quadratic_bregman_divergence(input, target):
    r"""
    The log quadratic bregman divergence.
    This is defined as D_F(exp(p), exp(q)), where D_F is the bregman divergence
    corresponding to the potential

    F(x):=(1-x) \text{Li}_2(1-x)-\frac{1}{2} x \log ^2(x)

    Explicitly, we have:
    L(x, y) := (e^x-1) \text{Li}_2(1-e^y)-(e^x-1)
        \text{Li}_2(1-e^x)+\frac{1}{2} e^x (y^2-x^2)
    where y is the input and x is the target.
    """
    quadratic_part = 0.5 * target.exp() * (input**2 - target**2)
    correction = torch.expm1(target) * (
        Spence1mExp.apply(input) - Spence1mExp.apply(target)
    )
    return quadratic_part + correction
