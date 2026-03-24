from math import sqrt

import torch


def integrate_sde(
    x0: float | torch.Tensor,
    drift: callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    a: float,
    n_steps: int = 1000,
):
    """
    Integrate the SDE dX= f(X,t)dt + \sqrt{2*a}dW, using Euler Mayurama Scheme

    Args:
        x0: the initial position
        drift: a function drift(x, t) defining the drift (f(X, t) in the equation above)
        a: the scale of the diffusive coefficient
        n_steps: The number of integration steps
    """
    x = x0
    dt = 1 / n_steps
    for t in torch.linspace(0, 1, n_steps + 1, dtype=x0.dtype, device=x0.device)[:-1]:
        dx = drift(x, t) * dt
        db = sqrt(2 * a * dt) * torch.randn_like(x)
        x = x + dx + db

    return x
