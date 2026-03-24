from math import ceil, log, sqrt

import lightning as L
import torch
from einops import rearrange
from torch import nn, optim
from torch.utils.data import IterableDataset

from diffusion_rl.algorithms.integration import integrate_sde
from diffusion_rl.losses.log_quadratic_bregman import log_quadratic_bregman_divergence


class OnPolicySMCDataset(IterableDataset):
    r"""
    Generates data from the model, integratig the
    SDE dX= f(X,t)dt + \sqrt{2*a}dW, using Euler Mayurama Scheme

    The generator yields batches of the form (x1, x, t)

    Args:
        dim: The ambient dimension (of x).
        drift: a function drift(x, t) defining the drift (f(X, t) in the equation above)
        value: a function value_fn(x, t) used to define the target (usually the actual value function you are training)
        smc_value: a function value_fn(x, t) to reweight samples via weights w_i=exp(value_fn(x_i, t)),
            for the purpose of SMC resampling
        a: the scale of the diffusive coefficient
        n_steps: The number of integration steps
        mc_samples_per_step: The number of samples drawn per step (for the SMC estimate)
    """

    def __init__(
        self,
        dim: int,
        drift: callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        value: callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        smc_value: callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        reward: callable[[torch.Tensor], torch.Tensor],
        device,
        sampling_method,
        a: float = 1,
        batch_size: int = 1024,
        n_steps: int = 1000,
        mc_samples_per_step: int = 10,
        lambda_eff=0.1,
    ):
        self.drift = drift
        self.value = value
        self.smc_value = smc_value
        self.reward = reward
        self.batch_size = batch_size
        self.mc_samples_per_step = mc_samples_per_step
        self.n_steps = n_steps
        self.dim = dim
        self.device = device
        self.a = a
        full_size = batch_size * mc_samples_per_step * n_steps
        self._x = torch.zeros((full_size, dim), device=device)
        self._t = torch.zeros((full_size, 1), device=device)
        self._y = torch.zeros((full_size, 1), device=device)
        self._loc = full_size
        self.sampling_method = sampling_method
        self.lambda_eff = lambda_eff

    def drift_fn(self, x, t):
        """This just computes the drift function on x extended by mc samples"""
        bs, mc, d = x.shape
        x_ = rearrange(x, "bs mc d -> (bs mc) d")
        dr = self.drift(x_, t)
        assert dr.shape == (bs * mc, d)
        return rearrange(dr, "(bs mc) d -> bs mc d", bs=bs, mc=mc)

    def smc_value_fn(self, x, t):
        """This just computes the value function on x extended by mc samples"""
        bs, mc, d = x.shape
        x_ = rearrange(x, "bs mc d -> (bs mc) d")
        v = self.smc_value(x_, t)
        assert v.shape == (bs * mc,)
        v = v.unsqueeze(-1)
        return rearrange(v, "(bs mc) d -> bs mc d", bs=bs, mc=mc)

    def value_fn(self, x, t):
        """This just computes the value function on x extended by mc samples"""
        bs, mc, d = x.shape
        x_ = rearrange(x, "bs mc d -> (bs mc) d")
        v = self.value(x_, t)
        assert v.shape == (bs * mc,)
        r = self.reward(x_)
        assert r.shape == (bs * mc,)
        t = t.squeeze(-1)
        v_out = t * r + (1 - t) * v
        v_out = v_out.unsqueeze(-1)
        assert v_out.shape == (bs * mc, 1)
        return rearrange(v_out, "(bs mc) d -> bs mc d", bs=bs, mc=mc)

    def __iter__(self):
        # This generator function runs in an infinite loop
        while True:
            if self._x is None or self._loc >= self._x.shape[0]:
                if self.sampling_method == "one_step_bootstrap":
                    all_x, all_t, all_tgt = one_step_bootstrap(
                        drift=self.drift_fn,  # (B, N, dim), (B, N, 1) -> (B, N, dim)
                        value=self.value_fn,  # log F(x,t): same sig -> (B*N, 1)
                        log_tau=self.smc_value_fn,  # log τ(x,t): same sig -> (B*N, 1)
                        h=self.reward,  # log h(x): (B, N, dim) -> (B, N, 1), or None to reuse value at t=1
                        a=self.a,  # diffusion coefficient
                        batch_size=ceil(self.batch_size / self.mc_samples_per_step),
                        mc_samples=self.mc_samples_per_step,
                        dim=self.dim,
                        n_steps=self.n_steps,
                        device=self.device,
                    )
                elif self.sampling_method == "ancestral_td_lambda":
                    with torch.no_grad():
                        all_x, all_t, all_tgt = ancestral_td_lambda(
                            drift=self.drift,  # (B*N, dim), (B*N, 1) -> (B*N, dim)
                            value=self.value,  # log F(x,t): same sig -> (B*N, 1)
                            log_tau=self.smc_value,  # log τ(x,t): same sig -> (B*N, 1)
                            h=self.reward,  # log h(x): (B*N, dim) -> (B*N, 1), or None to reuse value at t=1
                            a=self.a,  # diffusion coefficient
                            lambda_eff=self.lambda_eff,  # effective lambda = λ^n_steps  ∈ [0,1]
                            batch_size=ceil(self.batch_size / self.mc_samples_per_step),
                            mc_samples=self.mc_samples_per_step,
                            dim=self.dim,
                            n_steps=self.n_steps,
                            device=self.device,
                        )
                elif self.sampling_method == "single_seed_td_lambda":
                    with torch.no_grad():
                        all_x, all_t, all_tgt = single_seed_td_lambda(
                            drift=self.drift,  # (B*N, dim), (B*N, 1) -> (B*N, dim)
                            value=self.value,  # log F(x,t): same sig -> (B*N, 1)
                            log_tau=self.smc_value,  # log τ(x,t): same sig -> (B*N, 1)
                            h=self.reward,  # log h(x): (B*N, dim) -> (B*N, 1), or None to reuse value at t=1
                            a=self.a,  # diffusion coefficient
                            lambda_eff=self.lambda_eff,  # effective lambda = λ^n_steps  ∈ [0,1]
                            batch_size=self.batch_size,
                            mc_samples=self.mc_samples_per_step,
                            dim=self.dim,
                            n_steps=self.n_steps,
                            device=self.device,
                        )
                elif self.sampling_method == "single_seed_mc":
                    with torch.no_grad():
                        all_x, all_t, all_tgt = single_seed_mc(
                            drift=self.drift,  # (B*N, dim), (B*N, 1) -> (B*N, dim)
                            value=self.value,  # log F(x,t): same sig -> (B*N, 1)
                            log_tau=self.smc_value,  # log τ(x,t): same sig -> (B*N, 1)
                            h=self.reward,  # log h(x): (B*N, dim) -> (B*N, 1), or None to reuse value at t=1
                            a=self.a,  # diffusion coefficient
                            batch_size=self.batch_size,
                            mc_samples=self.mc_samples_per_step,
                            dim=self.dim,
                            n_steps=self.n_steps,
                            device=self.device,
                        )
                self._x = all_x
                self._t = all_t.unsqueeze(-1)
                self._y = all_tgt.unsqueeze(-1)
                self._loc = 0

            x = self._x[self._loc]
            y = self._y[self._loc]
            t = self._t[self._loc]
            yield y, x, t
            self._loc += 1


# define the LightningModule
_T_BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
_T_BIN_NAMES = ["t00_20", "t20_40", "t40_60", "t60_80", "t80_100"]


def _log_binned(module, prefix, values, t_flat):
    """Log per-t-bin statistics of `values` (shape N) keyed on t_flat (shape N)."""
    for name, lo, hi in zip(_T_BIN_NAMES, _T_BIN_EDGES[:-1], _T_BIN_EDGES[1:]):
        mask = (t_flat >= lo) & (t_flat < hi)
        if mask.sum() > 1:
            module.log(f"{prefix}_{name}", values[mask].var(),
                       on_step=True, on_epoch=False, prog_bar=False)


class OnPolicyValue(L.LightningModule):
    def __init__(
        self,
        base_score_module,
        value_module,
        a,
        lr,
        reward_function=None,
        dim: int = 2,
        loss_type: str = "mse",
        grad_decay: float = None,
        analytical_value_fn=None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["base_score_module", "value_module", "reward_function", "analytical_value_fn"]
        )
        self.reward_function = reward_function
        self.base_score_module = base_score_module
        self.value_module = value_module
        self.loss_type = loss_type
        self.a = a
        self.analytical_value_fn = analytical_value_fn

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        y, x, t = batch
        if self.hparams.grad_decay is not None:
            x = x.clone().detach().requires_grad_(True)
        pred_value = self.value_module(x, t.flatten()).flatten()[:, None]
        true_value = y.flatten()[:, None]
        if self.loss_type == "mse":
            loss = nn.functional.mse_loss(torch.exp(pred_value), torch.exp(true_value))
        elif self.loss_type == "quad":
            loss = log_quadratic_bregman_divergence(pred_value, true_value).mean()
        self.log("train_loss", loss)
        # Per-bin variance of (target - V_analytical), measuring training-target noise
        if self.analytical_value_fn is not None:
            with torch.no_grad():
                v_anal = self.analytical_value_fn(x.detach(), t.flatten())
                target_err = true_value.flatten() - v_anal
            _log_binned(self, "target_var", target_err, t.flatten())

        if self.hparams.grad_decay is not None:
            # Take the gradient of the value, and add an l2 decay to its magnitude

            #    - create_graph=True : keep the graph for higher‑order grads
            #    - retain_graph=True : we still need the graph for the next step
            value_grad = torch.autograd.grad(
                pred_value.sum(),
                x,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]  # shape same as inputs

            # Norm of the gradient
            grad_norm = value_grad.norm(p=2, dim=-1)  # (bs,)

            loss = loss + self.hparams.grad_decay * grad_norm.mean()

        return loss

    def sigma_2(self, t):
        """The diffusive variance at time t"""
        return 2 * self.a * t * (1 - t)

    def sigma(self, t):
        """The diffusive scale at time t"""
        return torch.sqrt(self.sigma_2(t))

    def drift(self, x: torch.Tensor, t: torch.Tensor, beta=1):
        r"""
        The SDE is dX = (u + v) dt + sqrt(2a) dW, X_0 = 0
        (constant diffusion coefficient, matching integrate_sde).

        The optimal control is  v = 2a * grad V(x, t)
        where V(x, t) = log E[exp(r(X_T)) | X_t = x].
        """
        with torch.inference_mode(False):
            x_clone = x.clone()
            x_clone.requires_grad_(True)
            if x_clone.grad is not None:
                x_clone.grad.zero_()
            value = self.value_module(x_clone, t).sum()
            value.backward()
            value_grad = x_clone.grad
        base_score = self.base_score_module(x, t)
        guidance = 2 * self.a * value_grad
        return base_score + guidance * beta

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        if self.reward_function is None:
            return
        n_samples = 512
        x0 = torch.zeros(n_samples, self.hparams.dim, device=self.device)
        x_final = integrate_sde(x0, drift=self.drift, a=self.a, n_steps=100)
        rewards = self.reward_function(x_final)
        self.log("val_reward_mean", rewards.mean())
        self.log("val_reward_std", rewards.std())
        self.log("val_reward_max", rewards.max())
        t0 = torch.zeros(n_samples, device=self.device)
        val_at_0 = self.value_module(x0, t0).mean()
        self.log("val_value_at_t0", val_at_0)

        # Per-bin MAE of V_model vs V_analytical on a random eval grid
        if self.analytical_value_fn is not None:
            n_eval = 512
            x_eval = torch.randn(n_eval, self.hparams.dim, device=self.device)
            t_eval = torch.rand(n_eval, device=self.device)
            with torch.no_grad():
                v_pred = self.value_module(x_eval, t_eval)
                v_anal = self.analytical_value_fn(x_eval, t_eval)
            err = v_pred - v_anal
            for name, lo, hi in zip(_T_BIN_NAMES, _T_BIN_EDGES[:-1], _T_BIN_EDGES[1:]):
                mask = (t_eval >= lo) & (t_eval < hi)
                if mask.sum() > 0:
                    self.log(f"val_mae_{name}", err[mask].abs().mean())
                    self.log(f"val_bias_{name}", err[mask].mean())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.train()
        x0, *_ = batch
        x = integrate_sde(x0, drift=self.drift, a=self.a)
        return x


def one_step_bootstrap(
    drift,  # (B*N, dim), (B*N, 1) -> (B*N, dim)
    value,  # log F(x,t): same sig -> (B*N, 1)
    log_tau,  # log τ(x,t): same sig -> (B*N, 1)
    h,  # log h(x): (B*N, dim) -> (B*N, 1), or None to reuse value at t=1
    a,  # diffusion coefficient
    batch_size,
    mc_samples,
    dim,
    n_steps,
    device,
    dtype=torch.float32,
):
    """
    One step bootstrap.

    Runs a single SMC sweep with `batch_size * mc_samples` particles from x=0.
    Each particle's value estimate v_i = log F(x*_i, t) is a valid log-target
    for its parent x_i (by the martingale property).  Resampling decides which
    particles to refine.


    Returns:
        xs:      (B*N*(n_steps), dim)   -- flattened particles before resampling
        ts:      (B*N*(n_steps),)       -- corresponding times
        log_tgts:(B*N*(n_steps),)       -- TD(λ) log-targets
    """
    with torch.no_grad():
        x = torch.zeros(batch_size, mc_samples, dim, device=device, dtype=dtype)
        t = torch.zeros((batch_size * mc_samples, 1), device=x.device, dtype=dtype)
        v_smc = log_tau(x, t)
        ix = torch.arange(mc_samples, device=x.device)[None, :, None].expand(
            (batch_size, mc_samples, 1)
        )
        dt = 1 / n_steps
        xs = [x]
        targets = []
        ts = [0]
        for _t in torch.linspace(0, 1, n_steps + 1, dtype=x.dtype)[:-1]:
            t = torch.full(
                (batch_size * mc_samples, 1), _t, device=x.device, dtype=dtype
            )
            dx = drift(x, t) * dt
            db = sqrt(2 * a * dt) * torch.randn_like(x)
            x_next = x + dx + db
            v_next = value(x_next, t) if _t < 1 else h(x_next)

            # Compute the targets using the values:
            # if ix = [2, 0, 0], then sample 0 was duplicated: x[1] and x[2] are both identical
            # First count how many duplicates there are:
            counts = torch.zeros_like(v_next)
            counts.scatter_add_(1, ix, torch.ones_like(v_next))
            # In our example, counts = [0, 2, 1]

            # Now compute the log-mean-exp of the duplicate values.
            exp_val_sum = torch.zeros_like(v_next)
            exp_val_sum.scatter_add_(1, ix, torch.exp(v_next - v_next.max()))
            exp_val_mean = exp_val_sum / counts
            target = torch.log(exp_val_mean) + v_next.max()
            # In our example, we have [nan, log( (exp(v[1])+exp(v[2]))/2 ), v[0]]
            # Target only lives on the support of ix, we pull it back to all the samples,
            target_scattered = torch.gather(target, 1, ix)
            targets.append(target_scattered)
            # In our example, we have [v[0], log( (exp(v[1])+exp(v[2]))/2 ), log( (exp(v[1])+exp(v[2]))/2 )]
            # Since x[0] was a singleton, while x[1] and x[2] were identical (but the chilren were not).

            # Resample the samples
            v_smc_next = log_tau(x_next, t)
            rel_weights = torch.exp(v_smc_next - v_smc)

            # Update ix, x, and v for next pass.
            ix = torch.multinomial(
                rel_weights.squeeze(-1),
                num_samples=mc_samples,
                replacement=True,
            ).unsqueeze(-1)
            x = torch.gather(x_next, 1, ix.expand_as(x))
            v_smc = torch.gather(v_smc_next, 1, ix)

            # append the x:
            xs.append(x)  # Some of these are duplicates
            ts.append(_t)
        all_x = rearrange(xs[:-1], "n bs mc d -> (n bs mc) d")
        all_t = rearrange(
            [torch.full_like(v_smc, t) for t in ts[:-1]],
            "n bs mc d -> (n bs mc) d",
        ).squeeze(-1)
        all_tgt = rearrange(targets, "n bs mc d -> (n bs mc) d").squeeze(-1)
        assert all_x.shape[0] == all_t.shape[0] == all_tgt.shape[0]
        assert all_x.ndim == 2
        assert all_t.ndim == 1 == all_tgt.ndim
    return all_x, all_t, all_tgt


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _flat(x, batch_size, mc_samples, dim):
    return x.reshape(batch_size * mc_samples, dim)


def _tvec(t_scalar, batch_size, mc_samples, dtype, device):
    return torch.full(
        (batch_size * mc_samples, 1), t_scalar, dtype=dtype, device=device
    )


def _sde_step(x_flat, drift, a, t_scalar, dt, batch_size, mc_samples, dim, device):
    dtype = x_flat.dtype
    t_vec = _tvec(t_scalar, batch_size, mc_samples, dtype, device)
    dx = drift(x_flat, t_vec) * dt
    db = sqrt(2.0 * a * dt) * torch.randn_like(x_flat)
    return x_flat + dx + db


def _log_mean_exp_by_ancestor(log_vals, ancestor_ix):
    """
    log_vals:    (B, N, 1)  -- values on child particles
    ancestor_ix: (B, N, 1)  -- index of parent for each child
    Returns:     (B, N, 1)  -- log-mean-exp over children, on ancestor support.
                               Entries with no children are -inf.
    """
    B, N, _ = log_vals.shape
    anc_max = torch.full(
        (B, N, 1), float("-inf"), dtype=log_vals.dtype, device=log_vals.device
    )
    anc_max.scatter_reduce_(1, ancestor_ix, log_vals, reduce="amax", include_self=False)
    anc_max_clamped = anc_max.clamp(min=-1e38)
    shifted = log_vals - torch.gather(anc_max_clamped, 1, ancestor_ix)
    exp_sum = torch.zeros(B, N, 1, dtype=log_vals.dtype, device=log_vals.device)
    exp_sum.scatter_add_(1, ancestor_ix, shifted.exp())
    counts = torch.zeros(B, N, 1, dtype=log_vals.dtype, device=log_vals.device)
    counts.scatter_add_(1, ancestor_ix, torch.ones_like(log_vals))
    valid = counts > 0
    log_mean = torch.where(
        valid,
        torch.log(exp_sum.clamp(min=1e-38) / counts.clamp(min=1.0)) + anc_max_clamped,
        torch.full_like(anc_max, float("-inf")),
    )
    return log_mean, counts  # (B, N, 1), (B, N, 1)


def _resample(log_w, x_next, log_tau_next, batch_size, mc_samples, dim):
    """
    log_w:        (B, N, 1)
    x_next:       (B, N, dim)
    log_tau_next: (B, N, 1)
    Returns resampled x, log_tau, and index tensor ix (B, N, 1).
    """
    log_w_stable = log_w - log_w.amax(dim=1, keepdim=True)
    ix = torch.multinomial(
        log_w_stable.squeeze(-1).exp(),
        num_samples=mc_samples,
        replacement=True,
    ).unsqueeze(-1)  # (B, N, 1)
    x_r = torch.gather(x_next, 1, ix.expand_as(x_next))
    log_tau_r = torch.gather(log_tau_next, 1, ix)
    return x_r, log_tau_r, ix


# ---------------------------------------------------------------------------
# Algorithm 1 – Ancestral TD(λ)
# ---------------------------------------------------------------------------


def ancestral_td_lambda(
    drift,  # (B*N, dim), (B*N, 1) -> (B*N, dim)
    value,  # log F(x,t): same sig -> (B*N, 1)
    log_tau,  # log τ(x,t): same sig -> (B*N, 1)
    h,  # log h(x): (B*N, dim) -> (B*N, 1), or None to reuse value at t=1
    a,  # diffusion coefficient
    lambda_eff,  # effective lambda = λ^n_steps  ∈ [0,1]
    batch_size,
    mc_samples,
    dim,
    n_steps,
    device,
    dtype=torch.float32,
):
    """
    Ancestral TD(λ).

    Runs a single SMC sweep with `batch_size * mc_samples` particles from x=0.
    Each particle's value estimate v_i = log F(x*_i, t) is a valid log-target
    for its parent x_i (by the martingale property).  Resampling decides which
    particles to refine; log_mean_exp_by_ancestor averages sibling estimates.

    λ per step = lambda_eff^(1/n_steps), so the terminal weight is lambda_eff
    regardless of n_steps.

    Returns:
        xs:      (B*N*(n_steps), dim)   -- flattened particles before resampling
        ts:      (B*N*(n_steps),)       -- corresponding times
        log_tgts:(B*N*(n_steps),)       -- TD(λ) log-targets
    """
    lam = lambda_eff ** (1.0 / n_steps)  # per-step lambda
    dt = 1.0 / n_steps
    N = mc_samples

    x = torch.zeros(batch_size, N, dim, dtype=dtype, device=device)

    # Storage for forward pass
    step_xs = []  # particles before resampling, (B, N, dim)
    step_vs = []  # log-values after step,       (B, N, 1)
    step_ixs = []  # resample indices,             (B, N, 1)

    log_tau_x = log_tau(
        _flat(x, batch_size, N, dim), _tvec(0.0, batch_size, N, dtype, device)
    ).reshape(batch_size, N, 1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    for step_idx, _t in enumerate(torch.linspace(0, 1, n_steps + 1, dtype=dtype)[:-1]):
        t_curr = float(_t)
        t_next = t_curr + dt

        x_flat = _flat(x, batch_size, N, dim)
        x_next_flat = _sde_step(
            x_flat, drift, a, t_curr, dt, batch_size, N, dim, device
        )
        x_next = x_next_flat.reshape(batch_size, N, dim)

        t_next_vec = _tvec(t_next, batch_size, N, dtype, device)

        # v_i = log F(x*_i, t_next)  or  log h(x*_i) at terminal
        if step_idx == n_steps - 1 and h is not None:
            v = h(x_next_flat).reshape(batch_size, N, 1)
        else:
            v = value(x_next_flat, t_next_vec).reshape(batch_size, N, 1)

        log_tau_next = log_tau(x_next_flat, t_next_vec).reshape(batch_size, N, 1)
        log_w = log_tau_next - log_tau_x

        step_xs.append(x.clone())
        step_vs.append(v)

        x, log_tau_x, ix = _resample(log_w, x_next, log_tau_next, batch_size, N, dim)
        step_ixs.append(ix)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    T = len(step_vs)
    targets = [None] * T

    # Last step: target is just v (no future to blend)
    target = step_vs[T - 1]  # (B, N, 1)
    targets[T - 1] = target

    for j in range(T - 2, -1, -1):
        v_j = step_vs[j]  # (B, N, 1)
        ix_j = step_ixs[j]  # (B, N, 1)

        # Aggregate future target to parent support
        m_j, counts = _log_mean_exp_by_ancestor(target, ix_j)  # (B, N, 1)

        # For childless particles fall back to v_j
        future_log = torch.where(counts > 0, m_j, v_j)

        # TD(λ) blend in log-sum-exp form:
        #   target = log( (1-λ)*exp(v_j) + λ*exp(future_log) )
        stacked = torch.stack([v_j + log(1 - lam), future_log + log(lam)], dim=0)
        target = torch.logsumexp(stacked, dim=0)
        targets[j] = target

    # ------------------------------------------------------------------
    # Flatten and return
    # ------------------------------------------------------------------
    all_x = torch.cat([s.reshape(batch_size * N, dim) for s in step_xs], dim=0)
    all_t = torch.cat(
        [
            torch.full((batch_size * N,), step_ts, dtype=dtype, device=device)
            for step_ts in [
                float(torch.linspace(0, 1, n_steps + 1)[i]) for i in range(T)
            ]
        ],
        dim=0,
    )
    all_tgt = torch.cat([t.reshape(batch_size * N) for t in targets], dim=0)

    return all_x, all_t, all_tgt


# ---------------------------------------------------------------------------
# Shared forward pass for single-seed algorithms
# ---------------------------------------------------------------------------


def _single_seed_forward(
    drift,
    value,
    log_tau,
    h,
    a,
    batch_size,
    mc_samples,
    dim,
    n_steps,
    device,
    dtype,
):
    """
    Runs the single-seed SMC forward pass common to both single-seed algorithms.

    At each step a BATCH of seeds x (B, dim) is propagated; n mc_samples are
    drawn from each seed independently, giving a (B, N, dim) array of proposals.

    Returns lists (length n_steps) of:
        xs          (B, dim)    -- seed particle before step
        ts          float       -- time of seed
        log_z_ratios (B,)       -- log(1/N sum_i w_i), the step's log-Z estimate
        log_mean_vs  (B,)       -- log(1/N sum_i exp(v_i) / tau(x*_i))
                                   bootstrap term at t_next
        log_taus     (B,)       -- log tau(x_seed, t_curr)
    and the final seed particles x (B, dim) at t=1.
    """
    dt = 1.0 / n_steps
    N = mc_samples

    # One seed per batch element
    x = torch.zeros(batch_size, dim, dtype=dtype, device=device)  # (B, dim)

    xs_list = []
    ts_list = []
    log_z_list = []
    log_mean_v_list = []
    log_tau_list = []

    log_tau_x = log_tau(
        x, torch.full((batch_size, 1), 0.0, dtype=dtype, device=device)
    ).reshape([-1, 1])  # (B, 1)

    for step_idx, _t in enumerate(torch.linspace(0, 1, n_steps + 1, dtype=dtype)[:-1]):
        t_curr = float(_t)
        t_next = t_curr + dt

        # Expand seed to (B, N, dim) and draw N proposals
        x_exp = x.unsqueeze(1).expand(batch_size, N, dim)  # (B, N, dim)
        x_exp_flat = x_exp.reshape(batch_size * N, dim)

        t_curr_vec = _tvec(t_curr, batch_size, N, dtype, device)
        dx = drift(x_exp_flat, t_curr_vec) * dt
        db = sqrt(2.0 * a * dt) * torch.randn_like(x_exp_flat)
        x_next_flat = x_exp_flat + dx + db  # (B*N, dim)
        x_next = x_next_flat.reshape(batch_size, N, dim)

        t_next_vec = _tvec(t_next, batch_size, N, dtype, device)
        log_tau_next = log_tau(x_next_flat, t_next_vec).reshape(batch_size, N, 1)

        # Incremental weights: w_i = tau(x*_i, t_next) / tau(x_seed, t_curr)
        # log_tau_x is (B,1); broadcast over N
        log_w = log_tau_next - log_tau_x.unsqueeze(1)  # (B, N, 1)

        # log Z ratio: log(1/N sum_i w_i)  shape (B,)
        log_z_ratio = torch.logsumexp(log_w.squeeze(-1), dim=1) - log(N)  # (B,)

        # Value at proposals
        if step_idx == n_steps - 1 and h is not None:
            v = h(x_next_flat).reshape(batch_size, N, 1)
        else:
            v = value(x_next_flat, t_next_vec).reshape(batch_size, N, 1)

        # Resample to get next seed
        log_w_stable = log_w - log_w.amax(dim=1, keepdim=True)
        ix = torch.multinomial(
            log_w_stable.squeeze(-1).exp(),
            num_samples=N,
            replacement=True,
        )  # (B, N)

        # Bootstrap term: log(1/N sum_i exp(v_i[ix]) / tau(x*_i[ix], t_next))
        #   = log_mean_exp over resampled particles of (v - log_tau_next)
        v_r = torch.gather(v.squeeze(-1), 1, ix)  # (B, N)
        lt_r = torch.gather(log_tau_next.squeeze(-1), 1, ix)  # (B, N)
        log_mean_v = torch.logsumexp(v_r - lt_r, dim=1) - log(N)  # (B,)

        # Advance seed: pick first resampled particle
        x_next_r = torch.gather(
            x_next, 1, ix.unsqueeze(-1).expand(batch_size, N, dim)
        )  # (B, N, dim)
        x = x_next_r[:, 0, :]  # (B, dim)

        log_tau_x = log_tau(
            x, torch.full((batch_size, 1), t_next, dtype=dtype, device=device)
        ).reshape([-1, 1])  # (B, 1)

        xs_list.append(x.clone())  # seed AFTER step (= x at t_next)
        ts_list.append(t_next)
        log_z_list.append(log_z_ratio)
        log_mean_v_list.append(log_mean_v)
        log_tau_list.append(log_tau_x.squeeze(-1))  # log tau at new seed

    return xs_list, ts_list, log_z_list, log_mean_v_list, log_tau_list


# ---------------------------------------------------------------------------
# Algorithm 2 – Single-Seed TD(λ)
# ---------------------------------------------------------------------------


def single_seed_td_lambda(
    drift,
    value,
    log_tau,
    h,
    a,
    lambda_eff,
    batch_size,
    mc_samples,
    dim,
    n_steps,
    device,
    dtype=torch.float32,
):
    """
    Single-Seed TD(λ).

    A single seed per batch element is propagated forward under the twisted
    chain; at each step N proposals are drawn to estimate the one-step
    bootstrap and the Z ratio.  TD(λ) blends k-step returns backward.

    λ per step = lambda_eff^(1/n_steps).

    Returns:
        xs:      (B * n_steps, dim)
        ts:      (B * n_steps,)
        log_tgts:(B * n_steps,)
    """
    lam = lambda_eff ** (1.0 / n_steps)

    xs_list, ts_list, log_z_list, log_mean_v_list, log_tau_list = _single_seed_forward(
        drift, value, log_tau, h, a, batch_size, mc_samples, dim, n_steps, device, dtype
    )

    T = n_steps
    # We compute targets for xs_list[0..T-1], which are seeds at t_1..t_T
    # Target at step j estimates H(x_{j-1}, t_{j-1}) -- the seed BEFORE the step.
    # However for simplicity we attach the target to the seed AFTER the step,
    # i.e. xs_list[j] at ts_list[j], matching the bootstrap definition:
    #   H(x, t_j) ~ tau(x, t_j) * 1/N sum_i F(x*_i, t_{j+1}) / tau(x*_i, t_{j+1})
    # which uses log_mean_v_list[j] and log_tau_list[j].

    # Initialise at last step: pure one-step bootstrap
    #   log_target = log_tau[T-1] + log_mean_v[T-1]
    log_target = log_tau_list[-1] + log_mean_v_list[-1]  # (B,)
    log_tau_curr = log_tau_list[-1]  # (B,)

    log_targets = [log_target]

    for j in range(T - 2, -1, -1):
        log_z = log_z_list[j + 1]  # Z ratio for step j -> j+1
        log_mv = log_mean_v_list[j]  # one-step bootstrap at j
        new_log_tau = log_tau_list[j]  # log tau at seed x_j

        # k-step estimate propagated from j+1 back to j:
        #   log H_hat^(k)(x_j) = new_log_tau + log_z + log_target - log_tau_curr
        log_k_step = new_log_tau + log_z + log_target - log_tau_curr

        # one-step estimate at j:
        #   log H_hat^(1)(x_j) = new_log_tau + log_mv
        log_1_step = new_log_tau + log_mv

        # TD(λ) blend:
        #   log_target = log( (1-λ)*exp(log_1_step) + λ*exp(log_k_step) )
        stacked = torch.stack([log_1_step + log(1 - lam), log_k_step + log(lam)], dim=0)
        log_target = torch.logsumexp(stacked, dim=0)  # (B,)
        log_tau_curr = new_log_tau
        log_targets.append(log_target)

    # log_targets was built back-to-front
    log_targets = log_targets[::-1]

    # Stack and return; xs_list[j] is the seed at ts_list[j]
    all_x = torch.stack(xs_list, dim=1).reshape(batch_size * T, dim)
    all_t = (
        torch.tensor(ts_list, dtype=dtype, device=device)
        .unsqueeze(0)
        .expand(batch_size, T)
        .reshape(batch_size * T)
    )
    all_tgt = torch.stack(log_targets, dim=1).reshape(batch_size * T)

    return all_x, all_t, all_tgt


# ---------------------------------------------------------------------------
# Algorithm 3 – Single-Seed Monte Carlo
# ---------------------------------------------------------------------------


def single_seed_mc(
    drift,
    value,
    log_tau,
    h,
    a,
    batch_size,
    mc_samples,
    dim,
    n_steps,
    device,
    dtype=torch.float32,
):
    """
    Single-Seed Monte Carlo.

    Same forward pass as Single-Seed TD(λ), but the backward pass telescopes
    the full Z-product rather than blending bootstrap estimates.  Equivalent
    to lambda_eff=1 in Single-Seed TD(λ) but written explicitly for clarity
    and without the log(0) issue at lam=1.

        log H_hat(x_j) = log_tau(x_j) + sum_{k=j}^{T-1} log_z_ratio_k
                        + log_mean_v_T

    Returns:
        xs:      (B * n_steps, dim)
        ts:      (B * n_steps,)
        log_tgts:(B * n_steps,)
    """
    xs_list, ts_list, log_z_list, log_mean_v_list, log_tau_list = _single_seed_forward(
        drift, value, log_tau, h, a, batch_size, mc_samples, dim, n_steps, device, dtype
    )

    T = n_steps

    # Initialise: terminal bootstrap
    log_target = log_tau_list[-1] + log_mean_v_list[-1]  # (B,)
    log_tau_curr = log_tau_list[-1]
    log_targets = [log_target]

    for j in range(T - 2, -1, -1):
        log_z = log_z_list[j + 1]
        new_log_tau = log_tau_list[j]

        # Telescope: multiply in the next Z factor and adjust tau
        #   log H_hat(x_j) = new_log_tau + log_z + log_target - log_tau_curr
        log_target = new_log_tau + log_z + log_target - log_tau_curr
        log_tau_curr = new_log_tau
        log_targets.append(log_target)

    log_targets = log_targets[::-1]

    all_x = torch.stack(xs_list, dim=1).reshape(batch_size * T, dim)
    all_t = (
        torch.tensor(ts_list, dtype=dtype, device=device)
        .unsqueeze(0)
        .expand(batch_size, T)
        .reshape(batch_size * T)
    )
    all_tgt = torch.stack(log_targets, dim=1).reshape(batch_size * T)

    return all_x, all_t, all_tgt
