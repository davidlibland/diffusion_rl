"""Plain squared error in LOG space: (p - q)**2.

This is the well-conditioned-but-BIASED baseline of the paper's section 3.
Its gradient 2(p - q) is linear in the error and never overflows or
vanishes, so it has none of the conditioning pathologies of the exp-space
losses.  What it does not have is the right minimiser: squared error in q is
minimised by E[q | x_t], whereas the value is V = log E[e^q | x_t], and the
two differ by the Jensen gap

    V - E[q] = log E[e^{q - E[q]}] >= 0,   ~ Var(q | x_t) / 2 for small spread,

which is largest exactly in the heavy-tailed, high-dimensional regime the
method is meant to handle.  It is included as an experimental arm to measure
that bias, not as a candidate loss.

Scale invariant in the same trivial sense as Itakura-Saito: it depends on
p and q only through p - q, so ``loss_shift`` has no effect.
"""

import torch


def log_mse(input: torch.Tensor, target: torch.Tensor):
    """Elementwise ``(input - target)**2``.

    No custom autograd Function is needed: the gradient 2(p - q) is finite
    for all finite inputs, so the standard autograd path is already stable.
    """
    d = input - target
    return d * d
