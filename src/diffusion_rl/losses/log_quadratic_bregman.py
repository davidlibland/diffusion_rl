import torch

from diffusion_rl.algorithms.spence import spence_1mexp_value

# Clamps used ONLY for the monitored loss VALUE (see _LogQuadraticBregman):
# inputs above _VALUE_CLAMP nats are pathological (exp overflows float32 at
# ~88), and the value is reported clamped rather than inf/nan.  The gradient
# is computed from the UNCLAMPED inputs via the closed form below.
_VALUE_CLAMP = 30.0
_VALUE_MAX = 1e30
# Final safety bound on the gradient: only reachable when the TRUE gradient
# overflows the floating-point range (e.g. exp(target) ~ inf), where training
# is already pathological; a huge-but-finite gradient keeps the optimizer
# state sane.
_GRAD_MAX = 1e30


def log_quadratic_bregman_grad(input: torch.Tensor, target: torch.Tensor):
    r"""Exact closed-form gradient of the log-quadratic Bregman divergence
    with respect to ``input`` (= p), holding ``target`` (= q) fixed:

        dL/dp = p * (e^p - e^q) / (e^p - 1)

    (The Spence terms telescope:  dL/dp = e^q*p + expm1(q)*S'(p) with
    S'(p) = -p*e^p/expm1(p) collapses to the expression above.)

    Computed branch-wise so it is finite for ALL float inputs:

      p < 0 :  p * (expm1(p) - expm1(q)) / expm1(p)
               (expm1(p) in [-1, 0); the expm1 difference is exact -- no
               cancellation -- and e^q underflow degrades gracefully to 0)
      p > 0 :  p * expm1(q - p) / expm1(-p)
               (factor e^p out of numerator and denominator: both expm1
               arguments are bounded above, so nothing overflows; for
               p -> inf this tends to the true asymptote  dL/dp -> p)
      p = 0 :  -expm1(q)        (the removable singularity's limit)

    The naive autograd of the loss NaNs for p >~ 88 in float32
    (Spence1mExp.backward evaluates e^p/expm1(p) = inf/inf) and returns
    -inf for q >~ 88 -- one prediction spike then poisons the weights and
    every subsequent loss is non-finite.
    """
    p, q = input, target
    g = torch.empty_like(p)

    mn = p < 0
    mp = p > 0
    m0 = p == 0

    if mn.any():
        em = torch.expm1(p[mn])
        g[mn] = p[mn] * (em - torch.expm1(q[mn])) / em
    if mp.any():
        g[mp] = p[mp] * torch.expm1(q[mp] - p[mp]) / torch.expm1(-p[mp])
    if m0.any():
        g[m0] = -torch.expm1(q[m0])

    # Only triggers when the true gradient exceeds the float range
    # (e.g. expm1(q) = inf for q > ~88 in float32).
    return torch.nan_to_num(g, nan=0.0, posinf=_GRAD_MAX, neginf=-_GRAD_MAX)


def _log_quadratic_bregman_value(input: torch.Tensor, target: torch.Tensor):
    """The raw divergence value (no stabilisation) -- reference/monitoring."""
    quadratic_part = 0.5 * target.exp() * (input**2 - target**2)
    correction = torch.expm1(target) * (
        spence_1mexp_value(input) - spence_1mexp_value(target)
    )
    return quadratic_part + correction


class _LogQuadraticBregman(torch.autograd.Function):
    """log-quadratic Bregman divergence with a stable, exact gradient.

    The loss VALUE is only used to monitor training, so the forward pass may
    sanitise it (clamp inputs to +-_VALUE_CLAMP nats, replace residual
    non-finite values): in the healthy regime (|inputs| < 30 nats) it is
    identical to the raw formula.  The BACKWARD pass ignores the sanitisation
    and returns the exact closed-form gradient at the true inputs
    (log_quadratic_bregman_grad), which is finite for all float inputs.
    """

    @staticmethod
    def forward(ctx, input, target):
        ctx.save_for_backward(input, target)
        # Clamp only the POSITIVE side: that is where exp/expm1 overflow.
        # The negative side is numerically safe at any magnitude (exp
        # underflows to 0, expm1 -> -1, Spence -> pi^2/6) and very negative
        # log-values are routine in this codebase, so clamping it would
        # distort the monitored value.
        val = _log_quadratic_bregman_value(
            input.clamp(max=_VALUE_CLAMP),
            target.clamp(max=_VALUE_CLAMP),
        )
        return torch.nan_to_num(val, nan=_VALUE_MAX, posinf=_VALUE_MAX,
                                neginf=-_VALUE_MAX)

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        if ctx.needs_input_grad[1]:
            raise NotImplementedError(
                "log_quadratic_bregman_divergence: gradient w.r.t. the target "
                "is not implemented -- regression targets must be detached "
                "(they are everywhere in this codebase)."
            )
        return grad_output * log_quadratic_bregman_grad(input, target), None


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

    Numerics: the gradient w.r.t. ``input`` is the exact closed form
    ``p*(e^p - e^q)/(e^p - 1)`` evaluated stably for all float inputs (the
    naive autograd path is NaN for input > ~88 and -inf for target > ~88 in
    float32).  The returned VALUE is sanitised for monitoring (inputs clamped
    to +-30 nats, non-finite results replaced) -- identical to the raw
    divergence in the healthy regime.  The target must be detached.
    """
    return _LogQuadraticBregman.apply(input, target)
