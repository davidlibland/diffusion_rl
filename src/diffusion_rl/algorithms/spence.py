import math

import torch


def spence_power_series(x: torch.Tensor, n_terms: int = 50) -> torch.Tensor:
    """
    Compute the Spence function Li_2(x) using the approximation
    Li_2(x)=sum_{k=1}^infty x^k/k^2

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (any shape). Must be real‑valued.
    n_terms : int, optional
        Number of terms used in the power‑series expansion for |x| < 1.
        Default 50 gives ≈ 1e‑12 accuracy for |x| ≤ 0.9.

    Returns
    -------
    torch.Tensor
        The value of Li_2(x) with the same shape as x.
    """
    # Build the series up to n_terms
    k = torch.arange(1, n_terms + 1, dtype=x.dtype, device=x.device)
    for _ in range(x.ndim):
        k = torch.unsqueeze(k, dim=0)
    # term_k = x^k / k^2
    # We broadcast x over k
    series_terms = (x.unsqueeze(-1) ** k) / (k**2)
    return series_terms.sum(dim=-1)


def spence_1mexp_value(x: torch.Tensor, n_terms: int = 100) -> torch.Tensor:
    """
    Computes Li_2(1-exp(x))

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (any shape). Must be real‑valued.

    Returns
    -------
    torch.Tensor
        The value of Li_2(1-exp(x)) with the same shape as x.
    """
    pi = math.pi
    pi2_over_6 = pi**2 / 6.0
    result = torch.zeros_like(x)
    # -----------------------------------------------------------------
    # a. Constant for |x| < -5
    # -----------------------------------------------------------------
    mask_series = x <= -5
    result[mask_series] = pi2_over_6

    # -----------------------------------------------------------------
    # b. Power‑series for |x| < 1   (Li_2(x) = Σ_{k=1}∞ x^k / k^2)
    # -----------------------------------------------------------------
    mask_series = (x < 0) & (x > -5)
    result[mask_series] = spence_power_series(1 - x[mask_series].exp(), n_terms=n_terms)

    # -----------------------------------------------------------------
    # c. Dedicated power series (of length 10) near 0
    # -----------------------------------------------------------------
    mask_series = (x >= 0) & (x < 3)
    y = x[mask_series] / 10
    result[mask_series] = (
        -10 * y
        - 25 * y**2
        - (250 * y**3) / 9
        + (250 * y**5) / 9
        - (62500 * y**7) / 1323
        + (156250 * y**9) / 1701
    )

    # -----------------------------------------------------------------
    # c. Dedicated power series (of length 10) near 5
    # -----------------------------------------------------------------
    mask_series = (x >= 3) & (x < 7)
    y = (x[mask_series] - 5) / 10
    result[mask_series] = (
        -14.1044
        - 50.3392 * y
        - 48.6318 * y**2
        - 3.49205 * y**3
        + 6.15862 * y**4
        - 7.50845 * y**5
        + 5.39981 * y**6
        + 0.451104 * y**7
        - 7.09654 * y**8
        + 9.60132 * y**9
        - 4.71555 * y**10
    )

    # ------------------------------------------------------------------
    # d. Analytic continuation for x > 1
    #      Li_2(x) = -Li_2(1/x) - π²/6 - 0.5 * ln(-x)²
    #      (same formula works for negative x because ln(-x) is real)
    #
    #  ln(exp(x)-1)=ln(exp(x)(1-exp(-x)))=x+ln1p(-exp(-x))
    # ------------------------------------------------------------------
    mask_series = x >= 7
    x_ = x[mask_series]
    inv1mexp = -torch.exp(-x_) / (torch.exp(-x_) - 1)
    lnexpm1 = x_ + torch.log1p(-torch.exp(-x_))
    result[mask_series] = (
        -spence_power_series(inv1mexp, n_terms=n_terms) - pi2_over_6 - 0.5 * lnexpm1**2
    )

    return result.to(x)


class Spence1mExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save the input of the forward pass to be used in the backward pass
        output = spence_1mexp_value(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input
        (input,) = ctx.saved_tensors
        # Compute the gradient: t*exp(t)/(1-exp(t))
        result = torch.zeros_like(input)

        # Avoid dividing by zero:
        mask_zero = input.abs() < 0.001
        input_ = input[mask_zero]
        result[mask_zero] = -(
            grad_output[mask_zero]
            * input_.exp()
            / (1 + input_ / 2 + input_**2 / 6 + input_**3 / 24 + input_**4 / 120)
        )

        # Rest of the values can use explicit formula:
        mask_zero = input.abs() >= 0.001
        input_ = input[mask_zero]
        result[mask_zero] = (
            -grad_output[mask_zero] * input_ * input_.exp() / torch.expm1(input_)
        )

        # Return gradients for all inputs (in this case, only one input)
        return result
