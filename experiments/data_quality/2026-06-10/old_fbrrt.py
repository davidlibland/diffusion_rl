"""
Frozen copy of the *pre-fix* (buggy) FBRRT-SMC estimators, extracted verbatim
from the on_policy.py at commit `master` (8c5782e) before the FBRRT FBSDE
backward-pass fixes (B, D, A/C).

Used by run_fbrrt_data_quality.py to compare the OLD vs NEW FBRRT target quality
on the data-quality benchmark.  Do not edit -- this is a historical reference.

Helpers that did not change (FBRRTSamples, _resample_fbrrt) are imported from the
live package so only the four estimator bodies are frozen here.
"""

from math import log, sqrt  # noqa: F401

import torch
from einops import rearrange
from torch import Tensor
from typing import Callable

from diffusion_rl.models.on_policy import FBRRTSamples, _resample_fbrrt

def fbrrt_smc_grad_control(
    *,
    a: float,
    n_steps: int,
    n_particles: int,
    branch: int,
    f: Callable[[Tensor, Tensor], Tensor],
    v_theta: Callable[[Tensor, Tensor], Tensor],
    reward: Callable[[Tensor], Tensor],
    d: int,
    alpha: float = 1.0,
    entropy_lambda: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    resample_method: str = "systematic",
) -> FBRRTSamples:
    """
    FBRRT-SMC where control = grad_x v_theta is derived automatically.

    This is the natural special case u*(x,t) = grad_x v_theta(x,t).
    No separate `control` argument is required.  Both the sampling drift
    K = f + alpha*2a * grad_x v_theta  and the BSDE driver correction
     -sqrt(2a) * grad_x v_theta  are computed from a single autograd
    call per parent node, reused across forward and backward passes.

    The forward SDE uses an interpolated drift:
        dX_t = (f + alpha * 2a * grad_x v_theta) dt + sqrt(2a) dW_t

    alpha=1 samples under the optimal (on-policy) drift; alpha=0 samples
    under the base drift f.

    The Girsanov correction from K back to f^mu = f + 2a*grad_x v is:
        D_t = (f^mu - K) / sqrt(2a) = (1 - alpha) * sqrt(2a) * grad_x v

    giving BSDE driver:
        -1/2 |Z|^2 + Z * D_t
        = -a|grad_x v|^2 + (1-alpha)*2a|grad_x v|^2
        = a(1 - 2*alpha) * |grad_x v|^2

    Special cases:
        alpha=1: driver = -a |grad_x v|^2 * dt          (on-policy, D_t=0)
        alpha=0: driver = +a |grad_x v|^2 * dt          (base drift)
        alpha=0.5: driver = 0
    Parameters
    ----------
    a               Diffusion constant  (dX uses sqrt(2a) dW).
    n_steps         N: number of Euler-Maruyama steps; dt = 1/N.
    n_particles     M: particle budget after each resampling step.
    branch          B: children sampled per particle before resampling.
    f               Base drift  f(x, t) -> [M, d].
    v_theta         Current value function  v_theta(x, t) -> [M].
    d               State-space dimension.
    x0              Initial state [d] or [M, d].  Defaults to zeros.
    entropy_lambda  Temperature for local-entropy reweighting.
                    inf -> uniform SMC weights.
    device          Torch device.
    dtype           Torch dtype.
    resample_method "systematic" (default, lower variance) or "multinomial".
    alpha           Drift interpolation in [0, 1].  Default 1.0 (on-policy).

    Returns
    -------
    FBRRTSamples  (x, t, v_hat, weights)
    """
    device = device or torch.device("cpu")
    dt = 1.0 / n_steps
    sqdt = dt**0.5
    sq2a = (2 * a) ** 0.5
    M, B, N = n_particles, branch, n_steps

    x = torch.zeros(M, d, device=device, dtype=dtype)

    step_data = []

    for i in range(N):
        t_i = i * dt
        t_next = (i + 1) * dt
        t_i_tensor = torch.tensor(t_i).to(x)
        t_next_tensor = torch.tensor(t_next).to(x)

        # -- grad_x v_theta at parents: used for both K and D --
        x_in = x.detach().requires_grad_(True)
        with torch.enable_grad():
            v_par = v_theta(x_in, t_i_tensor.expand(M))  # [M]
            grad_x_v = torch.autograd.grad(
                v_par.sum(),
                x_in,
                create_graph=False,
            )[0].detach()  # [M, d]
        parent_x = x_in.detach()

        with torch.no_grad():
            dW = torch.randn(M, B, d, device=device, dtype=dtype) * sqdt
            f_val = f(parent_x, t_i_tensor.expand((M, 1)))  # [M, d]
            K = f_val + alpha * 2 * a * grad_x_v  # [M, d]

            children = (
                parent_x.unsqueeze(1) + K.unsqueeze(1) * dt + sq2a * dW
            ).reshape(M * B, d)

            v_ch = v_theta(children, t_next_tensor.expand(M * B))  # [M*B]
            v_ch_mb = v_ch.reshape(M, B)

            log_w = (
                torch.zeros(M * B, device=device, dtype=dtype)
                if entropy_lambda == float("inf")
                else v_ch / entropy_lambda
            )
            log_w = log_w - log_w.logsumexp(dim=0)
            w_new = log_w.exp()

            indices = _resample_fbrrt(w_new, M, method=resample_method)
            x = children[indices]

        step_data.append(
            {
                "t_i": t_i,
                "parent_x": parent_x,  # [M, d]
                "grad_x_v": grad_x_v,  # [M, d]  = control at parent
                "v_ch_mb": v_ch_mb,  # [M, B]
                "w_flat": w_new.reshape(M, B),  # [M, B]
            }
        )

    # -- Backward pass --
    # Sampling drift K = f + alpha*2a*grad_x_v
    # So D_t = (f^mu - K) / sqrt(2a) = (1 - alpha) * sqrt(2a) * grad_x v
    # driver = a*(1 - 2*alpha) * |grad_x_v|^2 * dt
    driver_coeff = a * (1.0 - 2.0 * alpha)
    all_x, all_t, all_v_hat, all_w = [], [], [], []

    for data in step_data:
        t_i = data["t_i"]
        parent_x = data["parent_x"]
        grad_x_v = data["grad_x_v"]  # [M, d]
        v_ch_mb = data["v_ch_mb"]  # [M, B]
        w_flat = data["w_flat"]  # [M, B]

        w_norm = w_flat / w_flat.sum(dim=1, keepdim=True)
        ev_next = (w_norm * v_ch_mb).sum(dim=1)  # [M]

        driver = driver_coeff * (grad_x_v**2).sum(dim=-1) * dt  # [M]

        all_x.append(parent_x)
        all_t.append(torch.full((M,), t_i, device=device, dtype=dtype))
        all_v_hat.append((ev_next + driver).detach())
        all_w.append(torch.full((M,), 1.0 / M, device=device, dtype=dtype))

    # add reward targets at t=1:
    all_x.append(x)
    all_t.append(torch.full((M,), 1, device=device, dtype=dtype))
    all_v_hat.append(reward(x))
    all_w.append(torch.full((M,), 1.0 / M, device=device, dtype=dtype))

    return FBRRTSamples(
        x=rearrange(all_x, "M B d -> (M B) d"),
        t=rearrange(all_t, "M B -> (M B)"),
        v_hat=rearrange(all_v_hat, "M B -> (M B)"),
        weights=rearrange(all_w, "M B -> (M B)"),
    )


def fbrrt_smc_grad_control_td_lambda(
    *,
    a: float,
    n_steps: int,
    n_particles: int,
    branch: int,
    f: Callable[[Tensor, Tensor], Tensor],
    v_theta: Callable[[Tensor, Tensor], Tensor],
    reward: Callable[[Tensor], Tensor],
    d: int,
    lambda_eff: float = 0.95,
    alpha: float = 1.0,
    entropy_lambda: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    resample_method: str = "systematic",
) -> FBRRTSamples:
    """
    TD(lambda) version of fbrrt_smc_grad_control, with interpolated drift.

    Forward SDE:
        dX_t = (f + alpha * 2a * grad_x v_theta) dt + sqrt(2a) dW_t

    Girsanov correction back to f^mu = f + 2a*grad_x v:
        D_t = (1 - alpha) * sqrt(2a) * grad_x v

    BSDE driver (combining -1/2|Z|^2 + Z*D_t):
        delta_i = a*(1 - 2*alpha) * |grad_x v|^2 * dt

    GAE recursion:
        G_N = r(X_1)
        G_i = delta_i + EV_{i+1} + lam * (G_{i+1} - EV_{i+1})

    Special cases for alpha:
        alpha=1:   delta = -a|grad_x v|^2 * dt      (on-policy, D_t=0)
        alpha=0:   delta = +a|grad_x v|^2 * dt      (base drift)
        alpha=0.5: delta = 0                         (no driver)

    Special cases for lam:
        lam=0:  one-step bootstrap (equivalent to fbrrt_smc_grad_control)
        lam=1:  full Monte Carlo trajectory
    Parameters
    ----------
    lam             Lambda parameter in [0, 1].  Default 0.95.
    alpha           Drift interpolation in [0, 1].  Default 1.0 (on-policy).
    (all others)    Same as fbrrt_smc_grad_control.
    """
    lam = lambda_eff ** (1.0 / n_steps)  # per-step lambda
    device = device or torch.device("cpu")
    dt = 1.0 / n_steps
    sqdt = dt**0.5
    sq2a = (2 * a) ** 0.5
    M, B, N = n_particles, branch, n_steps

    x = torch.zeros(M, d, device=device, dtype=dtype)

    step_data = []

    # ------------------------------------------------------------------ #
    # Forward pass: identical to fbrrt_smc_grad_control                   #
    # ------------------------------------------------------------------ #
    for i in range(N):
        t_i = i * dt
        t_next = (i + 1) * dt
        t_i_tensor = torch.tensor(t_i, device=device, dtype=dtype).expand((M, 1))
        t_next_tensor = torch.tensor(t_next, device=device, dtype=dtype).expand(
            (M * B, 1)
        )

        x_in = x.detach().requires_grad_(True)
        with torch.enable_grad():
            v_par = v_theta(x_in, t_i_tensor)
            grad_x_v = torch.autograd.grad(
                v_par.sum(),
                x_in,
                create_graph=False,
            )[0].detach()
        parent_x = x_in.detach()

        with torch.no_grad():
            dW = torch.randn(M, B, d, device=device, dtype=dtype) * sqdt
            f_val = f(parent_x, t_i_tensor)
            K = f_val + alpha * 2 * a * grad_x_v  # interpolated drift

            children = (
                parent_x.unsqueeze(1) + K.unsqueeze(1) * dt + sq2a * dW
            ).reshape(M * B, d)

            v_ch = v_theta(children, t_next_tensor)
            v_ch_mb = v_ch.reshape(M, B)

            log_w = (
                torch.zeros(M * B, device=device, dtype=dtype)
                if entropy_lambda == float("inf")
                else v_ch / entropy_lambda
            )
            log_w = log_w - log_w.logsumexp(dim=0)
            w_new = log_w.exp()

            indices = _resample_fbrrt(w_new, M, method=resample_method)
            x = children[indices]

        step_data.append(
            {
                "t_i": t_i,
                "parent_x": parent_x,  # [M, d]
                "grad_x_v": grad_x_v,  # [M, d]
                "v_ch_mb": v_ch_mb,  # [M, B]
                "w_flat": w_new.reshape(M, B),  # [M, B]
            }
        )

    # ------------------------------------------------------------------ #
    # Backward pass: GAE-style lambda return                               #
    # ------------------------------------------------------------------ #
    # At each step i we have:
    #   EV_{i+1}  = weighted mean of v_theta over B children   [M]
    #   delta_i = a*(1 - 2*alpha) * |grad_x_v|^2 * dt [M]
    #
    # Recursion (sweep from i=N-1 down to i=0):
    #   G_N = r(X_1)  (terminal: x is the final particle positions)
    #   G_i = delta_i + EV_{i+1} + lam * (G_{i+1} - EV_{i+1})
    #
    # Note: G_{i+1} on the RHS refers to the target at the *resampled*
    # children, i.e. x_{i+1} = children[indices].  Since after resampling
    # we have M particles, G_{i+1} is also [M], aligned with x_{i+1}.

    # Terminal: reward at final particle positions
    t_terminal = torch.full((M,), 1.0, device=device, dtype=dtype)
    G = reward(x).detach()  # [M]
    driver_coeff = a * (1.0 - 2.0 * alpha)

    all_x = []
    all_t = []
    all_v_hat = []
    all_w = []

    for data in reversed(step_data):
        t_i = data["t_i"]
        parent_x = data["parent_x"]  # [M, d]
        grad_x_v = data["grad_x_v"]  # [M, d]
        v_ch_mb = data["v_ch_mb"]  # [M, B]
        w_flat = data["w_flat"]  # [M, B]

        # Weighted mean of v_theta over B children: EV_{i+1}
        w_norm = w_flat / w_flat.sum(dim=1, keepdim=True)  # [M, B]
        EV_next = (w_norm * v_ch_mb).sum(dim=1)  # [M]

        # BSDE driver: delta_i = a*(1 - 2*alpha) * |grad_x_v|^2 * dt
        delta = driver_coeff * (grad_x_v**2).sum(dim=-1) * dt  # [M]

        # GAE recursion:
        # G_i = delta_i + EV_{i+1} + lam * (G_{i+1} - EV_{i+1})
        G = delta + EV_next + lam * (G - EV_next)  # [M]

        all_x.append(parent_x)
        all_t.append(torch.full((M,), t_i, device=device, dtype=dtype))
        all_v_hat.append(G.detach())
        all_w.append(torch.full((M,), 1.0 / M, device=device, dtype=dtype))

    # Reverse to chronological order
    all_x.reverse()
    all_t.reverse()
    all_v_hat.reverse()
    all_w.reverse()

    # Append terminal reward targets
    all_x.append(x)
    all_t.append(t_terminal)
    all_v_hat.append(reward(x).detach())
    all_w.append(torch.full((M,), 1.0 / M, device=device, dtype=dtype))

    return FBRRTSamples(
        x=rearrange(all_x, "M B d -> (M B) d"),
        t=rearrange(all_t, "M B -> (M B)"),
        v_hat=rearrange(all_v_hat, "M B -> (M B)"),
        weights=rearrange(all_w, "M B -> (M B)"),
    )


def fbrrt_smc_grad_control_variate(
    *,
    a: float,
    n_steps: int,
    n_particles: int,
    branch: int,
    f: Callable[[Tensor, Tensor], Tensor],
    v_policy: Callable[[Tensor, Tensor], Tensor],
    v_target: Callable[[Tensor, Tensor], Tensor],
    reward: Callable[[Tensor], Tensor],
    d: int,
    alpha: float = 1.0,
    entropy_lambda: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    resample_method: str = "systematic",
) -> FBRRTSamples:
    """
    FBRRT-SMC with residual control variate Z estimator and separated
    policy / target value functions.

    Two value functions are accepted:

      v_policy  -- defines the SOC control u*(x,t) = grad_x v_policy.
                   Used to compute the sampling drift K and the Girsanov
                   correction D_t.  Freeze this (e.g. a lagged / EMA copy)
                   to stabilise exploration while v_target is being trained.

      v_target  -- defines the regression targets for V at each time step.
                   Used in the backward pass to compute the BSDE target
                       v_hat_i = E[V_target(X_{i+1})] + driver(Z_RCV)
                   This is the network being actively trained; its gradient
                   flows through the loss but NOT through the targets above.

    The Z estimator is the residual control variate

        Z_RCV = sigma^T grad_x v_policy(x_i)          <- low-variance anchor
              + (1/dt) * sum_b w_b * eps_b * dW_b      <- residual correction

    where eps_b = v_target(x_{i+1}^b) - v_policy(x_{i+1}^b) is the
    discrepancy between the two value functions at the children.  When
    v_target == v_policy (same weights, same network) eps = 0 and Z_RCV
    collapses to the pure gradient estimator.  As the two networks diverge,
    the residual correction provides an unbiased MRE-style correction with
    variance proportional to |eps|^2 / dt rather than |V|^2 / dt.

    The BSDE driver uses Z_RCV rather than grad_x v_policy:

        driver = a * [ -|z_rcv|^2 + 2*(1-alpha) * z_rcv . grad_x_v_policy ] * dt

    which recovers a*(1 - 2*alpha)*|grad_x v_policy|^2 * dt when eps -> 0.

    Forward SDE (sampling measure P):
        dX_t = (f + alpha * 2a * grad_x v_policy) dt + sqrt(2a) dW_t

    Girsanov correction (back to on-policy measure Q):
        D_t = (1 - alpha) * sqrt(2a) * grad_x v_policy

    Parameters
    ----------
    a               Diffusion constant (dX uses sqrt(2a) dW).
    n_steps         N: number of Euler-Maruyama steps; dt = 1/N.
    n_particles     M: particle budget after each resampling step.
    branch          B: children sampled per particle before resampling.
    f               Base drift  f(x, t) -> [M, d].
    v_policy        Value function defining the control / exploration drift.
                    Signature: (x: [M, d], t: [M]) -> [M].
                    Freeze this for stability while training v_target.
    v_target        Value function defining the regression targets.
                    Signature: (x: [M, d], t: [M]) -> [M].
                    Typically the network currently being optimised.
    reward          Terminal reward g(X_T) -> [M].
    d               State-space dimension.
    alpha           Drift interpolation in [0, 1].  Default 1.0 (on-policy).
    entropy_lambda  Temperature for local-entropy reweighting.
                    float('inf') -> uniform SMC weights.
    device          Torch device.
    dtype           Torch dtype.
    resample_method "systematic" (default) or "multinomial".

    Returns
    -------
    FBRRTSamples  (x, t, v_hat, weights)
    """
    device = device or torch.device("cpu")
    dt = 1.0 / n_steps
    sqdt = dt**0.5
    sq2a = (2 * a) ** 0.5
    M, B, N = n_particles, branch, n_steps

    x = torch.zeros(M, d, device=device, dtype=dtype)

    step_data: list[dict] = []

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    for i in range(N):
        t_i = i * dt
        t_next = (i + 1) * dt
        t_i_tensor = torch.tensor(t_i, device=device, dtype=dtype).expand(M)
        t_next_tensor = torch.tensor(t_next, device=device, dtype=dtype).expand(M * B)

        # grad_x v_policy at parents -- drives both K and D_t
        x_in = x.detach().requires_grad_(True)
        with torch.enable_grad():
            v_par_policy = v_policy(x_in, t_i_tensor)  # [M]
            grad_x_v_policy = torch.autograd.grad(
                v_par_policy.sum(), x_in, create_graph=False
            )[0].detach()  # [M, d]
        parent_x = x_in.detach()

        with torch.no_grad():
            dW = torch.randn(M, B, d, device=device, dtype=dtype) * sqdt  # [M, B, d]

            f_val = f(parent_x, t_i_tensor.unsqueeze(-1))  # [M, d]
            K = f_val + alpha * 2 * a * grad_x_v_policy  # [M, d]

            # Children positions: [M, B, d] -> [M*B, d]
            children = (
                parent_x.unsqueeze(1) + K.unsqueeze(1) * dt + sq2a * dW
            ).reshape(M * B, d)

            # ----------------------------------------------------------
            # Evaluate BOTH value functions at children (no grad needed)
            # v_policy at children: used for residual eps = v_target - v_policy
            # v_target at children: used for E[V_target(X_{i+1})] in target
            # ----------------------------------------------------------
            v_ch_policy = v_policy(children, t_next_tensor).reshape(M, B)  # [M, B]
            v_ch_target = v_target(children, t_next_tensor).reshape(M, B)  # [M, B]

            # SMC weights from v_target (the network being trained)
            log_w = (
                torch.zeros(M * B, device=device, dtype=dtype)
                if entropy_lambda == float("inf")
                else v_ch_target.reshape(M * B) / entropy_lambda
            )
            log_w = log_w - log_w.logsumexp(dim=0)
            w_new = log_w.exp()  # [M*B]

            indices = _resample_fbrrt(w_new, M, method=resample_method)
            x = children[indices]

        step_data.append(
            {
                "t_i": t_i,
                "t_next": t_next,
                "parent_x": parent_x,  # [M, d]
                "grad_x_v_policy": grad_x_v_policy,  # [M, d]
                "v_ch_policy": v_ch_policy,  # [M, B]  stop-grad
                "v_ch_target": v_ch_target,  # [M, B]  stop-grad
                "w_flat": w_new.reshape(M, B),
                "dW": dW,  # [M, B, d]
                "dt": dt,
            }
        )

    # ------------------------------------------------------------------
    # Backward pass  --  residual control variate for Z
    # ------------------------------------------------------------------
    # Sampling drift:   K = f + alpha * 2a * grad_x v_policy
    # On-policy drift:  f^mu = f + 2a * grad_x v_policy
    # Girsanov:         D_t = (f^mu - K) / sqrt(2a)
    #                       = (1 - alpha) * sqrt(2a) * grad_x v_policy
    #
    # Z_RCV = sqrt(2a) * [grad_x v_policy
    #                     + (1/dt) * sum_b w_b * eps_b * dW_b]
    #
    # where  eps_b = v_target(x_{i+1}^b) - v_policy(x_{i+1}^b)
    #
    # Driver  = -1/2 |Z|^2 dt + Z^T D_t dt
    #         = a * [-|z_rcv|^2 + 2*(1-alpha)* z_rcv . grad_x_v_policy] * dt
    #
    # (z_rcv is Z_RCV / sqrt(2a), i.e. the un-scaled version)
    # ------------------------------------------------------------------

    all_x, all_t, all_v_hat, all_w = [], [], [], []

    for data in step_data:
        t_i = data["t_i"]
        parent_x = data["parent_x"]  # [M, d]
        grad_x_v_policy = data["grad_x_v_policy"]  # [M, d]
        v_ch_policy = data["v_ch_policy"]  # [M, B]
        v_ch_target = data["v_ch_target"]  # [M, B]
        w_flat = data["w_flat"]  # [M, B]
        dW = data["dW"]  # [M, B, d]
        dt = data["dt"]

        w_norm = w_flat / w_flat.sum(dim=1, keepdim=True)  # [M, B], normalised

        with torch.no_grad():
            # Residual: discrepancy between the two value functions at children
            # Shape [M, B].  Zero when v_target and v_policy share weights.
            eps = v_ch_target - v_ch_policy  # [M, B]

            # Weighted residual correction to z:
            #   (1/dt) * sum_b w_b * eps_b * dW_b
            # [M, B, 1] * [M, B, d] -> sum over B -> [M, d]
            z_correction = (w_norm.unsqueeze(-1) * eps.unsqueeze(-1) * dW).sum(
                dim=1
            ) / dt  # [M, d]

        # Full z_rcv = grad_x v_policy + residual correction  [M, d]
        # grad_x_v_policy is already detached (no graph); z_correction
        # is also no_grad, so z_rcv carries no gradient -- targets are
        # stop-gradient by construction.
        z_rcv = grad_x_v_policy + z_correction  # [M, d]

        # Driver using Z_RCV
        #   a * [-|z_rcv|^2 + 2*(1-alpha) * z_rcv . grad_x_v_policy] * dt
        z_sq = (z_rcv**2).sum(dim=-1)  # [M]
        z_dot = (z_rcv * grad_x_v_policy).sum(dim=-1)  # [M]
        driver = a * (-z_sq + 2.0 * (1.0 - alpha) * z_dot) * dt  # [M]

        # Regression target: E_{w}[V_target(X_{i+1})] + driver(Z_RCV)
        ev_next = (w_norm * v_ch_target).sum(dim=1)  # [M]

        all_x.append(parent_x)
        all_t.append(torch.full((M,), t_i, device=device, dtype=dtype))
        all_v_hat.append((ev_next + driver).detach())
        all_w.append(torch.full((M,), 1.0 / M, device=device, dtype=dtype))

    # Terminal condition  (x is the resampled final population)
    all_x.append(x)
    all_t.append(torch.full((M,), 1.0, device=device, dtype=dtype))
    all_v_hat.append(reward(x).detach())
    all_w.append(torch.full((M,), 1.0 / M, device=device, dtype=dtype))

    return FBRRTSamples(
        x=rearrange(all_x, "N M d -> (N M) d"),
        t=rearrange(all_t, "N M -> (N M)"),
        v_hat=rearrange(all_v_hat, "N M -> (N M)"),
        weights=rearrange(all_w, "N M -> (N M)"),
    )


def fbrrt_smc_grad_mc_Z(
    *,
    a: float,
    n_steps: int,
    n_particles: int,
    branch: int,
    f: Callable[[Tensor, Tensor], Tensor],
    v_policy: Callable[[Tensor, Tensor], Tensor],
    v_target: Callable[[Tensor, Tensor], Tensor],
    reward: Callable[[Tensor], Tensor],
    d: int,
    alpha: float = 1.0,
    entropy_lambda: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    resample_method: str = "systematic",
) -> FBRRTSamples:
    """
    FBRRT-SMC with MC estiamte of Z and separated
    policy / target value functions.

    Two value functions are accepted:

      v_policy  -- defines the SOC control u*(x,t) = grad_x v_policy.
                   Used to compute the sampling drift K and the Girsanov
                   correction D_t.  Freeze this (e.g. a lagged / EMA copy)
                   to stabilise exploration while v_target is being trained.

    The Z estimator is the residual control variate

        Z = 1/dt * mean[ Y_{i+1} * B_{t_i,t_{i+1}}]

    where Y_{i+1}, B_{t_i,t_{i+1}} range over the children.

    The control is 2a* alpha* \grad v_policy:

    Forward SDE (sampling measure P):
        dX_t = (f + alpha * 2a * grad_x v_policy) dt + sqrt(2a) dW_t

    Girsanov correction (back to on-policy measure Q):
        D_t =  - alpha * sqrt(2a) * grad_x v_policy

    So that dY = -1/2 Z^2 dt + alpha * sqrt(2a) * Z * grad_x v_policy + Z dW

    Since Z = sqrt(2a) * grad_x v^*, for large sample sizes and
    a good v_policy ~= v^*, we have
    -1/2 Z^2 dt + alpha sqrt(2a) Z grad_x v_policy
    = -1/2 2a (\grad_x v^*)^2 + alpha * 2a (grad_x v^*)^2
    which vanishes when alpha = 1/2

    On the other hand, in practice there is an error in the estimate of Z.
    Let Z = \grad_x v^*+\eps and grad_x v_policy = \grad_x v^* +\eta. Then we have:
    -1/2 Z^2 dt + alpha * sqrt(2a) Z grad_x v_policy
    =-a (\grad_x v^*+\eps)^2 + alpha 2a (\grad_x v^*+\eps) (\grad_x v^* +\eta)
    = a * ( -\grad_x v^* ^2 - 2 \grad_x v^* \eps - \eps^2
            + 2 alpha \grad_x v^* ^2 + 2 alpha \grad_x v^* (\eps +\eta) + 2 alpha \eps \eta )
    = a * (
          (2 alpha-1) \grad_x v^* ^2
        + 2 (alpha-1) \grad_x v^* \eps
        + 2 alpha \grad_x v^* \eta
        + \eps ( 2 alpha \eta - \eps)
    So, to first order, the \eps term vanishes when \alpha = 1.


    Parameters
    ----------
    a               Diffusion constant (dX uses sqrt(2a) dW).
    n_steps         N: number of Euler-Maruyama steps; dt = 1/N.
    n_particles     M: particle budget after each resampling step.
    branch          B: children sampled per particle before resampling.
    f               Base drift  f(x, t) -> [M, d].
    v_policy        Value function defining the control / exploration drift.
                    Signature: (x: [M, d], t: [M]) -> [M].
    v_target        Value function defining the regression targets.
                    Signature: (x: [M, d], t: [M]) -> [M]. Used for the bootstrap target.
    reward          Terminal reward g(X_T) -> [M].
    d               State-space dimension.
    alpha           Drift interpolation in [0, 1].  Default 1.0 (on-policy).
    entropy_lambda  Temperature for local-entropy reweighting.
                    float('inf') -> uniform SMC weights.
    device          Torch device.
    dtype           Torch dtype.
    resample_method "systematic" (default) or "multinomial".

    Returns
    -------
    FBRRTSamples  (x, t, v_hat, weights)
    """
    device = device or torch.device("cpu")
    dt = 1.0 / n_steps
    sqdt = dt**0.5
    sq2a = (2 * a) ** 0.5
    M, B, N = n_particles, branch, n_steps

    x = torch.zeros(M, d, device=device, dtype=dtype)

    step_data: list[dict] = []

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    for i in range(N):
        t_i = i * dt
        t_next = (i + 1) * dt
        t_i_tensor = torch.tensor(t_i, device=device, dtype=dtype).expand(M)
        t_next_tensor = torch.tensor(t_next, device=device, dtype=dtype).expand(M * B)

        # grad_x v_policy at parents -- drives both K and D_t
        x_in = x.detach().requires_grad_(True)
        with torch.enable_grad():
            v_par_policy = v_policy(x_in, t_i_tensor)  # [M]
            grad_x_v_policy = torch.autograd.grad(
                v_par_policy.sum(), x_in, create_graph=False
            )[0].detach()  # [M, d]
        parent_x = x_in.detach()

        with torch.no_grad():
            dW = torch.randn(M, B, d, device=device, dtype=dtype) * sqdt  # [M, B, d]

            f_val = f(parent_x, t_i_tensor.unsqueeze(-1))  # [M, d]
            K = f_val + alpha * 2 * a * grad_x_v_policy  # [M, d]

            # Children positions: [M, B, d] -> [M*B, d]
            children = (
                parent_x.unsqueeze(1) + K.unsqueeze(1) * dt + sq2a * dW
            ).reshape(M * B, d)

            # ----------------------------------------------------------
            # Evaluate BOTH value functions at children (no grad needed)
            # v_policy at children: used for residual eps = v_target - v_policy
            # v_target at children: used for E[V_target(X_{i+1})] in target
            # ----------------------------------------------------------
            v_ch_policy = v_policy(children, t_next_tensor).reshape(M, B)  # [M, B]
            v_ch_target = v_target(children, t_next_tensor).reshape(M, B)  # [M, B]

            # SMC weights from v_target (the network being trained)
            log_w = (
                torch.zeros(M * B, device=device, dtype=dtype)
                if entropy_lambda == float("inf")
                else v_ch_policy.reshape(M * B) / entropy_lambda
            )
            log_w = log_w - log_w.logsumexp(dim=0)
            w_new = log_w.exp()  # [M*B]

            indices = _resample_fbrrt(w_new, M, method=resample_method)
            x = children[indices]

        step_data.append(
            {
                "t_i": t_i,
                "t_next": t_next,
                "parent_x": parent_x,  # [M, d]
                "grad_x_v_policy": grad_x_v_policy,  # [M, d]
                "v_ch_policy": v_ch_policy,  # [M, B]  stop-grad
                "v_ch_target": v_ch_target,  # [M, B]  stop-grad
                "w_flat": w_new.reshape(M, B),
                "dW": dW,  # [M, B, d]
                "dt": dt,
                "indices": indices,
                "children": children,
            }
        )

    # ------------------------------------------------------------------
    # Backward pass  --  residual control variate for Z
    # ------------------------------------------------------------------
    # Sampling drift:   K = f + alpha * 2a * grad_x v_policy
    # On-policy drift:  f^mu = f + 2a * grad_x v_policy
    # Girsanov:         D_t = (f^mu - K) / sqrt(2a)
    #                       = - alpha * sqrt(2a) * grad_x v_policy
    #
    # Z_i = (1/dt) * sum_b w_b * Y_{i+1} * dW_b
    #
    # Driver  = -1/2 |Z|^2 dt + Z^T D_t dt
    # ------------------------------------------------------------------

    y_mb = reward(children)  # [M*B]
    # Terminal condition: resampled final M particles + their reward target.
    # Keeping shapes [M, ...] to match the parent_x rows added in the loop.
    all_x, all_t, all_v_hat, all_w = (
        [x],
        [torch.full((M,), 1.0, device=device, dtype=dtype)],
        [reward(x).detach()],
        [torch.full((M,), 1.0 / M, device=device, dtype=dtype)],
    )
    y_m = None

    for data in reversed(step_data):
        t_i = data["t_i"]
        parent_x = data["parent_x"]  # [M, d]
        grad_x_v_policy = data["grad_x_v_policy"]  # [M, d]
        v_ch_target = data["v_ch_target"]  # [M, B]
        dW = data["dW"]  # [M, B, d]
        dt = data["dt"]
        ix = data["indices"]

        with torch.no_grad():
            if y_m is not None:
                # y_m is [M,] shaped, and lives on the ix indices.
                # We use v_ch_target to bootstrap the unselected children.
                y_mb = v_ch_target.reshape(M * B).clone()  # [M*B]
                y_mb[ix] = y_m

            m, br, _ = dW.shape  # m == M, br == B
            y_mb_ = rearrange(y_mb, "(m b) -> m b", m=m, b=br)  # [M, B]
            ydw = y_mb_.unsqueeze(-1) * dW  # [M, B, d]

            # MC estimate of Z via Z ~= 1/dt * E[Y dW]
            Z = ydw.mean(dim=1) / dt  # [M, d]

            y_m = (
                y_mb_.mean(dim=1)
                + (1 / 2 * (Z**2) - alpha * sqrt(2 * a) * (grad_x_v_policy * Z)).sum(
                    dim=1
                )
                * dt
            )  # [M]

        all_x.append(parent_x)
        all_t.append(torch.full((M,), t_i, device=device, dtype=dtype))
        all_v_hat.append((y_m).detach())
        all_w.append(torch.full((M,), 1.0 / M, device=device, dtype=dtype))

    return FBRRTSamples(
        x=rearrange(all_x, "N M d -> (N M) d"),
        t=rearrange(all_t, "N M -> (N M)"),
        v_hat=rearrange(all_v_hat, "N M -> (N M)"),
        weights=rearrange(all_w, "N M -> (N M)"),
    )
