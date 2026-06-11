"""MSE between exponentials of log-space inputs, with a stable exact gradient.

The naive ``(exp(p) - exp(q))**2`` overflows float32 once an input exceeds
~44 nats (the square needs exp(2p)), so training with loss_type="mse" was
skipping/aborting on every batch with a moderately large prediction (e.g.
18% of batches in the fbrrt_td_lambda BS=4 convergence run).

Same split as the log-quadratic Bregman loss: the VALUE is only used to
monitor training (computed with inputs clamped to +-_VALUE_CLAMP nats, then
sanitised), while the BACKWARD returns the exact closed-form gradient

    dL/dp = 2 * e^p * (e^p - e^q)

computed in the factored form  2 * e^p * e^p * (1 - e^{q-p})  and saturated
to +-_GRAD_MAX where the true gradient exceeds the float range (Adam
normalises per-coordinate, so a huge-but-finite gradient is a full-size,
correctly-signed step rather than weight poisoning).  Targets must be
detached.
"""

import torch

_VALUE_CLAMP = 30.0
_VALUE_MAX = 1e30
_GRAD_MAX = 1e30


def exp_mse_grad(input: torch.Tensor, target: torch.Tensor):
    """Exact gradient 2*e^p*(e^p - e^q), saturated to the float range.

    Factored as 2*(e^p)^2*(1 - e^{q-p}) so that:
      * p moderate, q very negative: -> 2*e^{2p}, exact;
      * p > ~44 (float32): e^{2p} = inf -> saturated to +-_GRAD_MAX with the
        correct sign from (1 - e^{q-p});
      * p very negative: e^p = 0 and the whole gradient correctly
        underflows to 0 (the 0 * inf = NaN case when ALSO q - p > ~88 is
        mapped to 0, matching the true limit -2*e^{p+q} -> 0 whenever the
        true gradient is representable).
    """
    p, q = input, target
    ep = torch.exp(p)
    g = 2.0 * ep * ep * (-torch.expm1(q - p))
    return torch.nan_to_num(g, nan=0.0, posinf=_GRAD_MAX, neginf=-_GRAD_MAX)


class _ExpMSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target):
        ctx.save_for_backward(input, target)
        # expm1 difference == exp difference exactly; clamp only the overflow
        # (positive) side so the monitored value stays faithful for the very
        # negative log-values that are routine here.
        d = (torch.expm1(input.clamp(max=_VALUE_CLAMP))
             - torch.expm1(target.clamp(max=_VALUE_CLAMP)))
        return torch.nan_to_num(d * d, nan=_VALUE_MAX, posinf=_VALUE_MAX)

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        if ctx.needs_input_grad[1]:
            raise NotImplementedError(
                "exp_mse: gradient w.r.t. the target is not implemented -- "
                "regression targets must be detached."
            )
        return grad_output * exp_mse_grad(input, target), None


def exp_mse(input, target):
    """Elementwise ``(exp(input) - exp(target))**2`` with a numerically
    stable exact gradient; see the module docstring."""
    return _ExpMSE.apply(input, target)
