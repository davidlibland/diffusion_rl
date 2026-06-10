from copy import deepcopy
from math import ceil, log, sqrt
from typing import Callable, NamedTuple

import lightning as L
import numpy as np
import torch
from einops import rearrange
from torch import Tensor, optim
from torch.utils.data import IterableDataset

from diffusion_rl.algorithms.integration import integrate_sde
from diffusion_rl.losses.log_quadratic_bregman import log_quadratic_bregman_divergence


class EMA:
    """
    Maintains an exponential moving average of a module's parameters.
    The EMA module is used as a stable target network for SMC sampling,
    while the live module is trained via gradient descent.
    """

    def __init__(self, module: torch.nn.Module, decay: float = 0.99):
        self.decay = decay
        self.shadow = deepcopy(module).to(next(module.parameters()).device)
        self.shadow.requires_grad_(False)

    @torch.no_grad()
    def update(self, module: torch.nn.Module):
        for shadow_p, live_p in zip(self.shadow.parameters(), module.parameters()):
            shadow_p.data.copy_(
                self.decay * shadow_p.data + (1.0 - self.decay) * live_p.data
            )

    def __call__(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)


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
        drift: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        value: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        smc_value: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        reward: Callable[[torch.Tensor], torch.Tensor],
        device,
        sampling_method,
        a: float = 1,
        batch_size: int = 1024,
        n_steps: int = 1000,
        mc_samples_per_step: int = 10,
        lambda_eff=0.1,
        branch=4,
        entropy_lambda=1.0,
        fbrrt_alpha=1.0,
        off_policy_frac: float = 0.0,
        generating_function: Callable[[int], "np.ndarray"] | None = None,
        random_t: bool = False,
        include_t_zero: bool = True,
        shuffle=True,
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
        # Lazy-allocate on-policy buffers on first use (not in __init__)
        # to avoid MPS memory pressure when off_policy_frac=1.0.
        self._x = None
        self._t = None
        self._y = None
        self._w = None
        self._loc = 0
        self.sampling_method = sampling_method
        self.lambda_eff = lambda_eff
        self.branch = branch
        self.entropy_lambda = entropy_lambda
        self.fbrrt_alpha = fbrrt_alpha

        # Off-policy mixing: fraction of samples drawn from forward-noised
        # base distribution with reward targets (stabilizing anchor).
        self.off_policy_frac = off_policy_frac
        self.generating_function = generating_function
        self.random_t = random_t
        self.include_t_zero = include_t_zero
        self.shuffle = shuffle

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

    def raw_value_fn(self, x, t):
        """Compute raw value function V(x, t) on (B, N, dim) input → (B, N, 1)."""
        bs, mc, d = x.shape
        x_ = rearrange(x, "bs mc d -> (bs mc) d")
        v = self.value(x_, t)
        assert v.shape == (bs * mc,)
        v = v.unsqueeze(-1)
        return rearrange(v, "(bs mc) d -> bs mc d", bs=bs, mc=mc)

    def value_fn(self, x, t):
        """Compute blended value t*r(x) + (1-t)*V(x,t) on (B, N, dim) input → (B, N, 1)."""
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

    def reward_fn(self, x):
        """Compute reward on (B, N, dim) shaped input, return (B, N, 1)."""
        bs, mc, d = x.shape
        x_ = rearrange(x, "bs mc d -> (bs mc) d")
        r = self.reward(x_)
        return rearrange(r.unsqueeze(-1), "(bs mc) d -> bs mc d", bs=bs, mc=mc)

    def __iter__(self):
        # This generator function runs in an infinite loop
        _first_on_policy = True
        while True:
            if self._x is None or self._loc >= self._x.shape[0]:
                # Free old buffers and flush the accelerator cache before
                # regenerating.  Guard on backend *availability*: torch.mps and
                # torch.cuda both expose empty_cache(), but calling the wrong
                # one (e.g. torch.mps.empty_cache() on a CUDA box) raises.
                if self._x is not None:
                    del self._x, self._t, self._y, self._w
                    self._x = self._t = self._y = self._w = None
                if _first_on_policy and str(self.device) != "cpu":
                    import gc

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    _first_on_policy = False
                # Per-sample regression weights.  Only the FBRRT methods produce
                # them (local-entropy LSMC weights); for all other methods they
                # stay None and the training loss falls back to an unweighted mean.
                all_weights = None
                if self.sampling_method == "one_step_bootstrap":
                    with torch.no_grad():
                        all_x, all_t, all_tgt = one_step_bootstrap(
                            drift=self.drift_fn,  # (B, N, dim), (B, N, 1) -> (B, N, dim)
                            value=self.raw_value_fn,  # log V(x,t): (B,N,dim) -> (B,N,1)
                            log_tau=self.smc_value_fn,  # log τ(x,t): same sig -> (B*N, 1)
                            h=self.reward_fn,  # log h(x): (B, N, dim) -> (B, N, 1)
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
                            random_t=self.random_t,
                            include_t_zero=self.include_t_zero,
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
                            random_t=self.random_t,
                            include_t_zero=self.include_t_zero,
                        )
                elif self.sampling_method == "ancestral_mc_td_lambda":
                    with torch.no_grad():
                        all_x, all_t, all_tgt = ancestral_mc_td_lambda(
                            drift=self.drift,  # (B*N, dim), (B*N, 1) -> (B*N, dim)
                            value=self.value,  # log F(x,t): same sig -> (B*N, 1)
                            log_tau=self.smc_value,  # log τ(x,t): same sig -> (B*N, 1)
                            h=self.reward,  # log h(x): (B*N, dim) -> (B*N, 1), or None to reuse value at t=1
                            a=self.a,  # diffusion coefficient
                            lambda_eff=self.lambda_eff,
                            batch_size=ceil(self.batch_size / self.mc_samples_per_step),
                            mc_samples=self.mc_samples_per_step,
                            dim=self.dim,
                            n_steps=self.n_steps,
                            device=self.device,
                        )
                elif self.sampling_method == "fbrrt":
                    with torch.no_grad():
                        all_x, all_t, all_tgt, all_weights = fbrrt_smc_grad_control(
                            a=self.a,
                            n_steps=self.n_steps,
                            n_particles=self.mc_samples_per_step,
                            branch=self.branch,
                            f=self.drift,
                            v_theta=self.value,
                            reward=self.reward,
                            d=self.dim,
                            entropy_lambda=self.entropy_lambda,
                            alpha=self.fbrrt_alpha,
                            device=self.device,
                        )
                elif self.sampling_method == "fbrrt_td_lambda":
                    with torch.no_grad():
                        all_x, all_t, all_tgt, all_weights = (
                            fbrrt_smc_grad_control_td_lambda(
                                a=self.a,
                                n_steps=self.n_steps,
                                n_particles=self.mc_samples_per_step,
                                branch=self.branch,
                                f=self.drift,
                                v_theta=self.value,
                                reward=self.reward,
                                d=self.dim,
                                lambda_eff=self.lambda_eff,
                                entropy_lambda=self.entropy_lambda,
                                alpha=self.fbrrt_alpha,
                                device=self.device,
                            )
                        )
                elif self.sampling_method == "fbrrt_cv":
                    with torch.no_grad():
                        all_x, all_t, all_tgt, all_weights = (
                            fbrrt_smc_grad_control_variate(
                                a=self.a,
                                n_steps=self.n_steps,
                                n_particles=self.mc_samples_per_step,
                                branch=self.branch,
                                f=self.drift,
                                v_policy=self.value,
                                v_target=self.value,
                                reward=self.reward,
                                d=self.dim,
                                entropy_lambda=self.entropy_lambda,
                                alpha=self.fbrrt_alpha,
                                device=self.device,
                            )
                        )
                elif self.sampling_method == "fbrrt_mc_z":
                    with torch.no_grad():
                        all_x, all_t, all_tgt, all_weights = fbrrt_smc_grad_mc_Z(
                            a=self.a,
                            n_steps=self.n_steps,
                            n_particles=self.mc_samples_per_step,
                            branch=self.branch,
                            f=self.drift,
                            v_policy=self.value,
                            v_target=self.value,
                            reward=self.reward,
                            d=self.dim,
                            entropy_lambda=self.entropy_lambda,
                            alpha=self.fbrrt_alpha,
                            device=self.device,
                        )
                # Splice in off-policy samples by overwriting a random subset of indices
                n_total = all_x.shape[0]
                if self.off_policy_frac > 0:
                    n_off = int(round(n_total * self.off_policy_frac))
                    off_idx = torch.randperm(n_total, device=self.device)[:n_off]

                    np_batch = self.generating_function(n_off)
                    x1 = torch.from_numpy(np_batch).to(
                        dtype=torch.float32, device=self.device
                    )
                    eps = torch.randn_like(x1)  # inherits device from x1
                    t_off = torch.rand(n_off, 1, device=self.device)
                    # NOTE: x_off is sampled from the *driftless* Brownian bridge
                    #   x_t = t*x1 + sqrt(2a t(1-t)) eps,  with target r(x1).
                    # Regressing exp(V) onto exp(r(x1)) is consistent ONLY if this
                    # matches the base process's law of (X_t | X_1=x1), i.e. ONLY
                    # if the base drift is zero (pure Brownian motion).  With a
                    # non-trivial base drift these anchors are off-distribution and
                    # their targets are biased.  Use off_policy_frac>0 only when the
                    # base process really is driftless BM (or supply a bridge that
                    # matches the actual base diffusion).
                    x_off = (
                        t_off * x1 + torch.sqrt(2 * self.a * t_off * (1 - t_off)) * eps
                    )
                    y_off = self.reward(x1)

                    all_x[off_idx] = x_off
                    all_t[off_idx] = t_off.reshape((n_off,) + all_t.shape[1:])
                    all_tgt[off_idx] = y_off.reshape((n_off,) + all_tgt.shape[1:])
                    # Off-policy anchors are ordinary (unweighted) regression
                    # samples: give them unit weight.
                    if all_weights is not None:
                        all_weights[off_idx] = 1.0
                # Default to uniform weights when the sampler did not supply any,
                # so every (x, t, y) row carries a weight for the loss.
                if all_weights is None:
                    all_weights = torch.ones(
                        n_total, device=all_x.device, dtype=all_x.dtype
                    )
                if self.shuffle:
                    perm = torch.randperm(n_total)
                else:
                    perm = torch.arange(n_total)
                self._x = all_x[perm]
                self._t = all_t[perm].unsqueeze(-1)
                self._y = all_tgt[perm].unsqueeze(-1)
                self._w = all_weights[perm].unsqueeze(-1)
                self._loc = 0

            else:
                x = self._x[self._loc]
                y = self._y[self._loc]
                t = self._t[self._loc]
                w = self._w[self._loc]
                yield y, x, t, w
                self._loc += 1


# define the LightningModule
_T_BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
_T_BIN_NAMES = ["t00_20", "t20_40", "t40_60", "t60_80", "t80_100"]


def _log_binned(module, prefix, values, t_flat):
    """Log per-t-bin statistics of `values` (shape N) keyed on t_flat (shape N)."""
    for name, lo, hi in zip(_T_BIN_NAMES, _T_BIN_EDGES[:-1], _T_BIN_EDGES[1:]):
        mask = (t_flat >= lo) & (t_flat < hi)
        if mask.sum() > 1:
            module.log(
                f"{prefix}_{name}",
                values[mask].var(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )


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
        ema_decay=0.99,  # EMA decay for stable SMC target
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "base_score_module",
                "value_module",
                "reward_function",
                "analytical_value_fn",
            ]
        )
        self.reward_function = reward_function
        self.base_score_module = base_score_module
        self.value_module = value_module
        self.loss_type = loss_type
        self.a = a
        self.analytical_value_fn = analytical_value_fn

        # EMA shadow network -- used as smc_value (frozen target)
        # Initialise EMA on CPU immediately -- will be moved to correct device
        # in setup() before any forward passes occur
        self.ema = EMA(value_module, decay=ema_decay)

    def _apply(self, fn, recurse=True):
        """Override for apply to ensure that it's called on the ema shadow too."""
        result = super()._apply(fn, recurse)
        # Keep EMA shadow on the same device as the live model.
        # _apply is the common path for .to(), .cpu(), .cuda(), etc.
        if hasattr(self, "ema"):
            self.ema.shadow._apply(fn, recurse)
        return result

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update EMA after every gradient step
        self.ema.update(self.value_module)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        # The dataset yields (y, x, t, w); w is the local-entropy LSMC regression
        # weight (uniform for non-FBRRT samplers).  Accept the legacy 3-tuple too.
        if len(batch) == 4:
            y, x, t, w = batch
        else:
            y, x, t = batch
            w = None
        if self.hparams.grad_decay is not None:
            x = x.clone().detach().requires_grad_(True)
        pred_value = self.value_module(x, t.flatten()).flatten()[:, None]
        true_value = y.flatten()[:, None]

        def _reduce(per_sample):
            # Weighted mean (Hawkins et al. 2020, eq. 23); plain mean if w is None.
            per_sample = per_sample.flatten()
            if w is None:
                return per_sample.mean()
            wf = w.flatten().clamp_min(0.0)
            return (wf * per_sample).sum() / wf.sum().clamp_min(1e-30)

        if self.loss_type == "mse":
            loss = _reduce((torch.exp(pred_value) - torch.exp(true_value)) ** 2)
        elif self.loss_type == "quad":
            loss = _reduce(log_quadratic_bregman_divergence(pred_value, true_value))
        self.log("train_loss", loss)
        if not torch.isfinite(loss).all():
            raise RuntimeError("Loss is not finite")
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

    def drift(self, x: torch.Tensor, t: torch.Tensor, beta=1, use_ema=True):
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
            if use_ema:
                value = self.ema(x_clone, t).sum()
            else:
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
    At each step the target for a particle is the log-mean-exp of V(children)
    across siblings sharing the same resampled parent (child-averaging).

    Includes both t=0 (all particles at origin) and t=1 (terminal, target=h).

    Returns:
        xs:      (B*N*(n_steps+1), dim)  -- particles at t=0, dt, ..., 1
        ts:      (B*N*(n_steps+1),)      -- corresponding times
        log_tgts:(B*N*(n_steps+1),)      -- one-step bootstrap log-targets
    """
    with torch.no_grad():
        BN = batch_size * mc_samples
        x = torch.zeros(batch_size, mc_samples, dim, device=device, dtype=dtype)
        t_vec = torch.zeros((BN, 1), device=device, dtype=dtype)
        v_smc = log_tau(x, t_vec)
        ix = torch.arange(mc_samples, device=device)[None, :, None].expand(
            (batch_size, mc_samples, 1)
        )
        dt = 1.0 / n_steps

        all_xs = []  # (B, N, dim)
        all_ts = []  # float
        all_tgts = []  # (B, N, 1)

        for step_idx, _t in enumerate(
            torch.linspace(0, 1, n_steps + 1, dtype=dtype)[:-1]
        ):
            t_curr = float(_t)
            t_next_scalar = t_curr + dt
            t_vec = torch.full((BN, 1), t_curr, device=device, dtype=dtype)
            t_next_vec = torch.full((BN, 1), t_next_scalar, device=device, dtype=dtype)

            # SDE step
            dx = drift(x, t_vec) * dt
            db = sqrt(2 * a * dt) * torch.randn_like(x)
            x_next = x + dx + db

            # Value of children (use h at terminal step)
            is_terminal = step_idx == n_steps - 1
            if is_terminal and h is not None:
                v_next = h(x_next)
            else:
                v_next = value(x_next, t_next_vec)

            # Child-average: log_mean_exp of v_next over siblings sharing
            # the same parent (tracked by ix from previous resampling)
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
            # In our example, we have [v[0], log( (exp(v[1])+exp(v[2]))/2 ), log( (exp(v[1])+exp(v[2]))/2 )]
            # Since x[0] was a singleton, while x[1] and x[2] were identical (but the chilren were not).

            # Store (x, t_curr, target) — particle x is at time t_curr
            all_xs.append(x)
            all_ts.append(t_curr)
            all_tgts.append(target_scattered)

            # Resample for next step
            v_smc_next = log_tau(x_next, t_next_vec)
            rel_weights = torch.exp(v_smc_next - v_smc)
            ix = torch.multinomial(
                rel_weights.squeeze(-1),
                num_samples=mc_samples,
                replacement=True,
            ).unsqueeze(-1)
            x = torch.gather(x_next, 1, ix.expand_as(x))
            v_smc = torch.gather(v_smc_next, 1, ix)

        # Terminal generation: x is post-resample at t=1, target = h(x)
        if h is not None:
            h_terminal = h(x)
        else:
            t1_vec = torch.full((BN, 1), 1.0, device=device, dtype=dtype)
            h_terminal = value(x, t1_vec)
        all_xs.append(x)
        all_ts.append(1.0)
        all_tgts.append(h_terminal)

        all_x = rearrange(all_xs, "n bs mc d -> (n bs mc) d")
        all_t = torch.cat(
            [torch.full((BN,), t, dtype=dtype, device=device) for t in all_ts]
        )
        all_tgt = rearrange(all_tgts, "n bs mc d -> (n bs mc) d").squeeze(-1)
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
    t_vec = _tvec(t_scalar, batch_size, mc_samples, x_flat.dtype, device)
    dx = drift(x_flat, t_vec) * dt
    db = sqrt(2.0 * a * dt) * torch.randn_like(x_flat)
    return x_flat + dx + db


def _log_mean_exp_by_ancestor(log_vals, ancestor_ix):
    """
    log_vals:    (B, N, 1)  -- values on child particles
    ancestor_ix: (B, N, 1)  -- index of parent for each child
    Returns:
        log_mean: (B, N, 1) -- log-mean-exp over children, on ancestor support
                               (-inf where no children)
        counts:   (B, N, 1) -- number of children per ancestor
    """
    B, N, _ = log_vals.shape
    anc_max = torch.full(
        (B, N, 1), float("-inf"), dtype=log_vals.dtype, device=log_vals.device
    )
    anc_max.scatter_reduce_(1, ancestor_ix, log_vals, reduce="amax", include_self=False)
    anc_max_c = anc_max.clamp(min=-1e38)
    shifted = log_vals - torch.gather(anc_max_c, 1, ancestor_ix)
    exp_sum = torch.zeros(B, N, 1, dtype=log_vals.dtype, device=log_vals.device)
    exp_sum.scatter_add_(1, ancestor_ix, shifted.exp())
    counts = torch.zeros(B, N, 1, dtype=log_vals.dtype, device=log_vals.device)
    counts.scatter_add_(1, ancestor_ix, torch.ones_like(log_vals))
    has_children = counts > 0
    log_mean = torch.where(
        has_children,
        torch.log(exp_sum.clamp(1e-38) / counts.clamp(1.0)) + anc_max_c,
        torch.full_like(anc_max, float("-inf")),
    )
    return log_mean, counts


def _avg_over_duplicates(log_vals, ix):
    """Average log_vals across particles that share the same resampling source.

    ix: (B, N, 1) — resampling indices that created the current particles.
         Particles i and j are duplicates iff ix[b, i] == ix[b, j].

    Returns (B, N, 1) with all duplicates holding the same log-mean-exp value.
    """
    avg_on_source, _ = _log_mean_exp_by_ancestor(log_vals, ix)
    return torch.gather(avg_on_source, 1, ix)


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


def _log_td_blend(log_one_step, log_multi_step, lam):
    """
    Computes log((1-λ)*exp(O) + λ*exp(M)) stably via logsumexp.
    All inputs broadcast-compatible.

    Special cases:
      lam=0 → returns log_one_step  (pure one-step bootstrap)
      lam=1 → returns log_multi_step (pure multi-step / MC)
    These avoid log(0) and ensure exact limiting behaviour.
    """
    if lam == 0.0:
        return log_one_step
    if lam == 1.0:
        return log_multi_step
    log_1m_lam = log(1.0 - lam)
    log_lam = log(lam)
    return torch.logsumexp(
        torch.stack([log_one_step + log_1m_lam, log_multi_step + log_lam], dim=0),
        dim=0,
    )


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

    Runs a single SMC sweep with batch_size * mc_samples particles from x=0.
    Each particle's value estimate v_i = log F(x*_i, t) is a valid log-target
    for its parent x_i (by the martingale property).  Resampling decides which
    particles to refine; log_mean_exp_by_ancestor averages sibling estimates.

    λ per step = lambda_eff^(1/n_steps), so the terminal weight is lambda_eff
    regardless of n_steps.

    The t=0 generation (all particles at x=0) is excluded from the output to
    keep the temporal range consistent with other sampling methods (dt to
    (n_steps-1)*dt).

    Returns:
        all_x:   (B*N*(n_steps-1), dim)   -- particles at t = dt, 2*dt, ..., (T-1)*dt
        all_t:   (B*N*(n_steps-1),)
        all_tgt: (B*N*(n_steps-1),)       -- log-targets for H(x,t)
    """
    lam = lambda_eff ** (1.0 / n_steps)  # per-step lambda
    dt = 1.0 / n_steps
    N = mc_samples

    x = torch.zeros(batch_size, N, dim, dtype=dtype, device=device)
    log_tau_x = log_tau(
        _flat(x, batch_size, N, dim),
        _tvec(0.0, batch_size, N, dtype, device),
    ).reshape(batch_size, N, 1)

    step_xs = []  # (B, N, dim) particles before resampling
    step_vs = []  # (B, N, 1)  log F at next-step proposals
    step_ixs = []  # (B, N, 1)  resample indices

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

        # v_i = log F(x*_i, t_next) or log h(x*_i) at terminal
        if step_idx == n_steps - 1 and h is not None:
            v = h(x_next_flat).reshape(batch_size, N, 1)
        else:
            v = value(x_next_flat, t_next_vec).reshape(batch_size, N, 1)

        log_tau_next = log_tau(x_next_flat, t_next_vec).reshape(batch_size, N, 1)
        log_w = log_tau_next - log_tau_x

        # Average v across duplicate particles created by previous resampling.
        # Duplicates share the same (x, t) but got independent SDE noise,
        # so their children's V values are independent samples of
        # E[V(x_next) | x].  Averaging reduces variance and at λ=0
        # recovers the one_step_bootstrap child-averaged target.
        if step_idx > 0:
            v = _avg_over_duplicates(v, step_ixs[-1])

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

        # Aggregate future target to parent support via log-mean-exp
        m_j, counts = _log_mean_exp_by_ancestor(target, ix_j)  # (B, N, 1)

        # Fallback for childless particles: use v_j (pure one-step)
        log_multi = torch.where(counts > 0, m_j, v_j)

        # TD(λ): log((1-λ)*exp(v_j) + λ*exp(log_multi))
        target = _log_td_blend(v_j, log_multi, lam)
        targets[j] = target

    # ------------------------------------------------------------------
    # Terminal generation (t=1): post-resample particles with h(x) target
    # ------------------------------------------------------------------
    x_terminal = x  # post-resample after last step
    if h is not None:
        tgt_terminal = h(_flat(x_terminal, batch_size, N, dim)).reshape(
            batch_size, N, 1
        )
    else:
        tgt_terminal = value(
            _flat(x_terminal, batch_size, N, dim),
            _tvec(1.0, batch_size, N, dtype, device),
        ).reshape(batch_size, N, 1)

    # ------------------------------------------------------------------
    # Flatten and return — include t=0 and t=1
    # ------------------------------------------------------------------
    ts_scalar = [float(torch.linspace(0, 1, n_steps + 1)[i]) for i in range(T)]

    all_x = torch.cat(
        [s.reshape(batch_size * N, dim) for s in step_xs]
        + [x_terminal.reshape(batch_size * N, dim)],
        dim=0,
    )
    all_t = torch.cat(
        [
            torch.full((batch_size * N,), ts_scalar[i], dtype=dtype, device=device)
            for i in range(T)
        ]
        + [torch.full((batch_size * N,), 1.0, dtype=dtype, device=device)],
        dim=0,
    )
    all_tgt = torch.cat(
        [t.reshape(batch_size * N) for t in targets]
        + [tgt_terminal.reshape(batch_size * N)],
        dim=0,
    )
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
    random_t: bool = False,
):
    """
    Single-seed SMC forward pass shared by both single-seed algorithms.

    At each step k (k=0..n_steps-1) a batch of seeds x (B, dim) at time
    ``t_k`` is propagated to ``t_{k+1}``; ``mc_samples`` proposals are
    drawn from each seed independently giving (B, N, dim).

    Time grid: if ``random_t`` is False (default), uniform with
    ``dt = 1/n_steps``. If True, the grid is ``[0, sorted U(0,1) draws
    (n_steps-1 of them), 1]``, so step sizes vary but endpoints are still
    0 and 1.

    Returns:
        xs_list:          length n_steps + 1.  ``xs_list[j]`` is the seed
                          at ``t_grid[j]``.  Index 0 is the initial seed
                          (zeros) at t=0; index n_steps is the terminal
                          seed at t=1.
        ts_list:          length n_steps + 1.  ``ts_list[j] = t_grid[j]``.
        log_z_list:       length n_steps.  ``log_z_list[k]`` is the log Z
                          ratio for step k (transition t_k → t_{k+1}).
        log_mean_v_list:  length n_steps.  ``log_mean_v_list[k]`` is the
                          bootstrap term ``log(1/N sum_i exp(v_i - log_tau_i))``
                          built from step k's proposals at ``t_{k+1}``,
                          which estimates V at the *pre-step* seed
                          ``xs_list[k]`` at time ``t_k``.  ``v_i`` is
                          ``h(x*_i)`` at the terminal step (when ``h`` is
                          provided), else ``value(x*_i, t_{k+1})``.
        log_tau_list:     length n_steps + 1.  ``log_tau_list[j]`` is
                          ``log tau(xs_list[j], t_grid[j])``.
    """
    N = mc_samples

    if random_t:
        inner = torch.rand(max(n_steps - 1, 0), dtype=dtype).sort().values
        t_grid = torch.cat(
            [
                torch.zeros(1, dtype=dtype),
                inner,
                torch.ones(1, dtype=dtype),
            ]
        )
    else:
        t_grid = torch.linspace(0, 1, n_steps + 1, dtype=dtype)

    x = torch.zeros(batch_size, dim, dtype=dtype, device=device)

    xs_list = []
    ts_list = []
    log_z_list = []
    log_mean_v_list = []
    log_tau_list = []

    log_tau_x = log_tau(
        x,
        torch.full((batch_size, 1), float(t_grid[0]), dtype=dtype, device=device),
    ).reshape([-1, 1])  # (B, 1) — log tau at the initial seed (t=0)

    for step_idx in range(n_steps):
        t_curr = float(t_grid[step_idx])
        t_next = float(t_grid[step_idx + 1])
        dt = t_next - t_curr

        # Record PRE-step seed and its log_tau (paired with log_mean_v below).
        xs_list.append(x.clone())
        ts_list.append(t_curr)
        log_tau_list.append(log_tau_x.squeeze(-1))

        # Expand seed to (B, N, dim) and draw N proposals at t_next.
        x_exp = x.unsqueeze(1).expand(batch_size, N, dim)
        x_exp_flat = x_exp.reshape(batch_size * N, dim)

        t_curr_vec = _tvec(t_curr, batch_size, N, dtype, device)
        dx = drift(x_exp_flat, t_curr_vec) * dt
        db = sqrt(2.0 * a * dt) * torch.randn_like(x_exp_flat)
        x_next_flat = x_exp_flat + dx + db  # (B*N, dim)
        x_next = x_next_flat.reshape(batch_size, N, dim)

        t_next_vec = _tvec(t_next, batch_size, N, dtype, device)
        log_tau_next = log_tau(x_next_flat, t_next_vec).reshape(batch_size, N, 1)

        # Incremental weights: w_i = tau(x*_i, t_next) / tau(x_seed, t_curr)
        log_w = log_tau_next - log_tau_x.unsqueeze(1)  # (B, N, 1)
        log_z_ratio = torch.logsumexp(log_w.squeeze(-1), dim=1) - log(N)  # (B,)

        # Value at proposals (h at terminal step if provided).
        is_terminal = step_idx == n_steps - 1
        if is_terminal and h is not None:
            v = h(x_next_flat).reshape(batch_size, N, 1)
        else:
            v = value(x_next_flat, t_next_vec).reshape(batch_size, N, 1)

        # Resample.
        log_w_stable = log_w - log_w.amax(dim=1, keepdim=True)
        ix = torch.multinomial(
            log_w_stable.squeeze(-1).exp(),
            num_samples=N,
            replacement=True,
        )  # (B, N)

        # Bootstrap term for step k, estimating V at the *pre-step* seed.
        v_r = torch.gather(v.squeeze(-1), 1, ix)  # (B, N)
        lt_r = torch.gather(log_tau_next.squeeze(-1), 1, ix)  # (B, N)
        log_mean_v = torch.logsumexp(v_r - lt_r, dim=1) - log(N)  # (B,)

        log_z_list.append(log_z_ratio)
        log_mean_v_list.append(log_mean_v)

        # Advance seed: pick first resampled particle.
        x_next_r = torch.gather(
            x_next, 1, ix.unsqueeze(-1).expand(batch_size, N, dim)
        )  # (B, N, dim)
        x = x_next_r[:, 0, :]  # (B, dim)

        log_tau_x = log_tau(
            x,
            torch.full((batch_size, 1), t_next, dtype=dtype, device=device),
        ).reshape([-1, 1])  # (B, 1)

    # Append the terminal seed (post final step) at t=1.
    xs_list.append(x.clone())
    ts_list.append(float(t_grid[-1]))
    log_tau_list.append(log_tau_x.squeeze(-1))

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
    random_t: bool = False,
    include_t_zero: bool = True,
):
    """
    Single-Seed TD(λ).

    A single seed per batch element propagates forward under the twisted
    chain; at each step N proposals estimate the one-step bootstrap and
    the Z ratio.  TD(λ) blends k-step returns backward via:

        log_target = log( (1-λ)*exp(one_step) + λ*exp(multi_step) )

    where, for the pre-step seed x_j at t_j:
        one_step_j  = log_tau(x_j, t_j) + log_mean_v[j]
        multi_step_j = log_tau(x_j, t_j) + log_z[j]
                       - log_tau(x_{j+1}, t_{j+1}) + log_target[j+1]

    λ per step = lambda_eff^(1/n_steps).

    NOTE: mc_samples proposals are used internally at each step but only
    ONE seed particle is kept per batch element per step.

    CAVEAT: the one-step term `log_tau(x_j,t_j) + log_mean_v[j]` is a
    self-normalised ratio estimator that is consistent only when `tau` is a
    martingale of the base diffusion (E_base[tau(X')|x] = tau(x)), i.e. as
    `tau -> H`.  With `tau` an EMA approximation of H it carries an O(tau - H)
    bias for `lambda < 1`.  The pure-MC limit (`single_seed_mc`, lambda_eff=1)
    and the child-averaged `one_step_bootstrap` do NOT have this dependence; the
    bias is the usual TD bootstrap bias and vanishes at convergence.

    Returns:
        all_x:   (batch_size * (n_steps + 1), dim)
                 seeds at t = 0, t_1, ..., t_{n_steps-1}, 1
        all_t:   (batch_size * (n_steps + 1),)
        all_tgt: (batch_size * (n_steps + 1),)
    """
    lam = lambda_eff ** (1.0 / n_steps)

    xs_list, ts_list, log_z_list, log_mean_v_list, log_tau_list = _single_seed_forward(
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
        random_t=random_t,
    )

    T = n_steps

    # Exact terminal target at t=1.
    x_terminal = xs_list[T]
    if h is not None:
        terminal_target = h(x_terminal).reshape(batch_size)
    else:
        t_terminal_vec = torch.full(
            (batch_size, 1), 1.0, dtype=dtype, device=device
        )
        terminal_target = value(x_terminal, t_terminal_vec).reshape(batch_size)

    # Backward blend over pre-step seeds at t_0..t_{T-1}.
    log_target = terminal_target  # seed the recursion with the exact terminal
    log_targets = []
    for j in range(T - 1, -1, -1):
        new_log_tau = log_tau_list[j]
        log_one_step = new_log_tau + log_mean_v_list[j]
        log_multi_step = (
            new_log_tau + log_z_list[j] + log_target - log_tau_list[j + 1]
        )
        log_target = _log_td_blend(log_one_step, log_multi_step, lam)
        log_targets.append(log_target)

    log_targets = log_targets[::-1]  # length T, paired with xs_list[0..T-1]
    log_targets.append(terminal_target)

    # Optionally drop the t=0 endpoint (degenerate point since x_0 is always
    # the same initial seed; with high-variance labels this can destabilize
    # training).
    if not include_t_zero:
        xs_list = xs_list[1:]
        ts_list = ts_list[1:]
        log_targets = log_targets[1:]

    T_out = len(xs_list)
    all_x = torch.stack(xs_list, dim=1).reshape(batch_size * T_out, dim)
    all_t = (
        torch.tensor(ts_list, dtype=dtype, device=device)
        .unsqueeze(0)
        .expand(batch_size, T_out)
        .reshape(batch_size * T_out)
    )
    all_tgt = torch.stack(log_targets, dim=1).reshape(batch_size * T_out)
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
    random_t: bool = False,
    include_t_zero: bool = True,
):
    """
    Single-Seed Monte Carlo.

    Same forward pass as Single-Seed TD(λ) but the backward pass telescopes
    the full Z-product (equivalent to lambda_eff=1 but without log(0) issues).

    For j in [0, n_steps-1]:
        log H_hat(x_j, t_j) = log_tau(x_j, t_j)
                            + sum_{k=j}^{T-1} log_z_ratio_k
                            + log_mean_v_{T-1}
    And at the terminal time t=1:
        log H_hat(x_T, 1) = h(x_T)         (exact)

    NOTE: mc_samples proposals are used internally at each step but only
    ONE seed particle is kept per batch element per step.

    Returns:
        all_x:   (batch_size * (n_steps + 1), dim)
                 seeds at t = 0, t_1, ..., t_{n_steps-1}, 1
        all_t:   (batch_size * (n_steps + 1),)
        all_tgt: (batch_size * (n_steps + 1),)
    """
    xs_list, ts_list, log_z_list, log_mean_v_list, log_tau_list = _single_seed_forward(
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
        random_t=random_t,
    )

    T = n_steps

    # Backward telescoping over pre-step seeds at t_0..t_{T-1}.
    #   target[T-1] = log_tau[T-1] + log_z[T-1] + log_mean_v[T-1]
    #   target[j]   = log_tau[j]   + log_z[j]   + target[j+1] - log_tau[j+1]
    log_target = log_tau_list[T - 1] + log_z_list[T - 1] + log_mean_v_list[T - 1]
    log_targets = [log_target]

    for j in range(T - 2, -1, -1):
        log_target = (
            log_tau_list[j] + log_z_list[j] + log_target - log_tau_list[j + 1]
        )
        log_targets.append(log_target)

    log_targets = log_targets[::-1]  # length T, paired with xs_list[0..T-1]

    # Exact terminal target: V(x_T, t=1) = h(x_T).
    x_terminal = xs_list[T]
    if h is not None:
        terminal_target = h(x_terminal).reshape(batch_size)
    else:
        t_terminal_vec = torch.full(
            (batch_size, 1), 1.0, dtype=dtype, device=device
        )
        terminal_target = value(x_terminal, t_terminal_vec).reshape(batch_size)
    log_targets.append(terminal_target)

    # Optionally drop the t=0 endpoint (degenerate point since x_0 is always
    # the same initial seed; with high-variance labels this can destabilize
    # training).
    if not include_t_zero:
        xs_list = xs_list[1:]
        ts_list = ts_list[1:]
        log_targets = log_targets[1:]

    T_out = len(xs_list)
    all_x = torch.stack(xs_list, dim=1).reshape(batch_size * T_out, dim)
    all_t = (
        torch.tensor(ts_list, dtype=dtype, device=device)
        .unsqueeze(0)
        .expand(batch_size, T_out)
        .reshape(batch_size * T_out)
    )
    all_tgt = torch.stack(log_targets, dim=1).reshape(batch_size * T_out)
    return all_x, all_t, all_tgt


# ---------------------------------------------------------------------------
# Algorithm 4 – Ancestral MC-TD(λ)
# ---------------------------------------------------------------------------


def ancestral_mc_td_lambda(
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
    Ancestral MC-TD(λ).

    Runs a standard SMC sweep (batch_size * mc_samples particles, resampled
    each step using τ-weights), then walks backward through the resampling
    tree to assign TD(λ) targets.

    The TD(λ) blend is performed in linear space before logging:
        R_i = log( (1-λ)*exp(O_i) + λ*exp(M_i) )
    where:
        O_i = V_i - log_tau(x_i, t)           one-step bootstrap
        M_i = log mean_{c in children(i)} [ w_c * rho_hat(c) ]   multi-step return

    with w_c = tau(c, t+dt)/tau(x_i, t) the child's incremental weight and
    rho_hat(c) = H(c, t+dt)/tau(c, t+dt) its downstream return estimate (the
    mean over c's resampled copies).  M_i is a mean of PRODUCTS over the
    children of x_i -- NOT a product of means: the recursion is
    rho(x,t) = E_q[w(x') rho(x', t+dt) | x_t = x], a single expectation of a
    product, so the weight and the return must be multiplied per child before
    averaging.  children(i) are the copies sharing x_i as resampling source,
    grouped by the resample indices that *created* x_i.  Finally
    target_i = R_i + log_tau(x_i, t).

    The t=0 generation (all particles at x=0) is NOT included in the output;
    the earliest stored generation is the post-resample particles at t=dt.
    The final post-resample generation at t=1 IS included (with exact reward
    targets), giving n_steps generations total.

    Returns:
        all_x:   (batch_size * mc_samples * n_steps, dim)
                 particles at t = dt, 2*dt, ..., n_steps*dt=1
        all_t:   (batch_size * mc_samples * n_steps,)
        all_tgt: (batch_size * mc_samples * n_steps,)
    """
    lam = lambda_eff ** (1.0 / n_steps)
    dt = 1.0 / n_steps
    N = mc_samples
    BN = batch_size * N

    def flat(z):
        return z.reshape(BN, dim)

    def tvec(t_scalar):
        return torch.full((BN, 1), t_scalar, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    x = torch.zeros(batch_size, N, dim, dtype=dtype, device=device)
    log_tau_x = log_tau(flat(x), tvec(0.0)).reshape(batch_size, N, 1)

    fwd_x_post = []  # (B, N, dim) post-resample at each step
    fwd_x_pre = []  # (B, N, dim) pre-resample (= post-resample of prev step)
    fwd_log_v = []  # (B, N, 1)  value at pre-resample particles of NEXT step
    fwd_log_w = []  # (B, N, 1)  log weights at pre-resample particles
    fwd_ix = []  # (B, N, 1)  resample indices
    fwd_ts = []  # float

    for step_idx, _t in enumerate(torch.linspace(0, 1, n_steps + 1, dtype=dtype)[:-1]):
        t_curr = float(_t)
        t_next = t_curr + dt

        x_flat = flat(x)
        x_next_flat = _sde_step(
            x_flat, drift, a, t_curr, dt, batch_size, N, dim, device
        )
        x_next = x_next_flat.reshape(batch_size, N, dim)

        log_tau_next = log_tau(x_next_flat, tvec(t_next)).reshape(batch_size, N, 1)
        log_w = log_tau_next - log_tau_x

        is_terminal = step_idx == n_steps - 1
        if is_terminal and h is not None:
            log_v = h(x_next_flat).reshape(batch_size, N, 1)
        else:
            log_v = value(x_next_flat, tvec(t_next)).reshape(batch_size, N, 1)

        # Average log_v across duplicate particles from previous resampling.
        # Same rationale as ancestral_td_lambda: duplicates share the same
        # (x, t) but their children had independent SDE noise.
        if step_idx > 0:
            log_v = _avg_over_duplicates(log_v, fwd_ix[-1])

        fwd_x_pre.append(x.clone())
        fwd_log_w.append(log_w)
        fwd_log_v.append(log_v)

        x_post, log_tau_x, ix = _resample(
            log_w, x_next, log_tau_next, batch_size, N, dim
        )

        fwd_x_post.append(x_post.clone())
        fwd_ix.append(ix)
        fwd_ts.append(t_next)

        x = x_post

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    # Base case: final post-resample generation
    #   R_i = h(x_i) - log_tau(x_i)  =>  target = h(x_i)
    x_final = fwd_x_post[-1]
    log_tau_final = log_tau(flat(x_final), tvec(fwd_ts[-1])).reshape(batch_size, N, 1)
    log_h_final = h(flat(x_final)).reshape(batch_size, N, 1)
    R = log_h_final - log_tau_final  # (B, N, 1)

    all_x_list = [flat(x_final)]
    all_t_list = [torch.full((BN,), fwd_ts[-1], dtype=dtype, device=device)]
    all_tgt_list = [(R + log_tau_final).reshape(BN)]

    for gen in range(n_steps - 1, 0, -1):
        # Two distinct resample-index tensors are needed here, on two different
        # index spaces (both length N, which is why mixing them up is a silent
        # bug):
        #   ix_next = fwd_ix[gen]   maps post-resample slots@gen -> child index@gen.
        #             R lives on the post-resample cloud x_post[gen], so this is
        #             how we aggregate a child's copies into its return estimate.
        #   ix_prev = fwd_ix[gen-1] maps slots@gen-1 -> child index@gen-1, i.e. it
        #             groups the copies of each distinct parent of generation gen-1
        #             (the same grouping used to duplicate-average V below).
        # log_w_gen is indexed by the *pre-resample* children of gen (one per
        # parent slot), aligned with x_post_prev: child p == SDE(x_post_prev[p]).
        ix_next = fwd_ix[gen]  # (B, N, 1) slots@gen -> child index@gen
        ix_prev = fwd_ix[gen - 1]  # (B, N, 1) copies of each distinct parent@gen-1
        log_w_gen = fwd_log_w[gen]  # (B, N, 1) weight of child p (parent slot p)
        x_post_prev = fwd_x_post[gen - 1]  # (B, N, dim) parents @ t_{gen-1}

        log_tau_post_prev = log_tau(flat(x_post_prev), tvec(fwd_ts[gen - 1])).reshape(
            batch_size, N, 1
        )

        # One-step term: O_i = V_i - log_tau(x_i, t_gen).
        # V = fwd_log_v[gen] is value at the child of slot i, already averaged
        # across a parent's copies during the forward pass (so O is constant
        # across copies of the same distinct parent). ✓
        V = fwd_log_v[gen]  # (B, N, 1)
        O = V - log_tau_post_prev  # (B, N, 1)

        # Per-child downstream return rho_hat(child_p) = H(child_p)/tau(child_p),
        # obtained by averaging R over the child's resampled copies.  R lives on
        # x_post[gen], so we group by ix_next; the result is indexed by child p.
        # has_desc[p] is False iff child p was never resampled (no descendants).
        log_mean_R, has_desc = _log_mean_exp_by_ancestor(R, ix_next)  # on child index

        # Per-child PRODUCT  w(child_p) * rho_hat(child_p).  This is a mean-of-
        # products, NOT a product-of-means: the recursion is
        #   rho(x,t) = E_q[ w(x') rho(x', t+dt) | x_t = x ],
        # a single expectation of the product w*rho, so w and rho must be
        # multiplied per child before averaging.  A childless child falls back to
        # its one-step value; the product then collapses to exp(O_p) (the same
        # w*rho with rho taken from value(child_p) instead of its descendants).
        log_prod = torch.where(has_desc > 0, log_w_gen + log_mean_R, O)  # (B, N, 1)

        # Multi-step term: M_i = log mean over i's children of the product.
        # i's children are the copies of distinct parent i, i.e. the slots sharing
        # i as resampling source -- grouped by ix_prev.  Average the products on
        # the parent's source support, then scatter back to every copy's slot so
        # duplicate slots receive the same (lower-variance) multi-step return.
        M_src, _ = _log_mean_exp_by_ancestor(log_prod, ix_prev)
        M = torch.gather(M_src, 1, ix_prev)  # (B, N, 1)

        # TD(λ) blend in linear space: R = log( (1-λ)exp(O) + λ exp(M) ).
        # Every slot has at least one child (itself), so M is always defined --
        # no childless fallback needed at this level (it is handled per child
        # in log_prod above).
        R = _log_td_blend(O, M, lam)  # (B, N, 1)

        target = (R + log_tau_post_prev).reshape(BN)

        all_x_list.append(flat(x_post_prev))
        all_t_list.append(
            torch.full((BN,), fwd_ts[gen - 1], dtype=dtype, device=device)
        )
        all_tgt_list.append(target)

    # ------------------------------------------------------------------
    # t=0 generation: initial particles (all at origin)
    # Use the backward-propagated R from gen=1 to compute the target
    # for the initial particles, just like the other generations.
    # ------------------------------------------------------------------
    ix_next0 = fwd_ix[0]  # slots@0 -> child index@0 (aggregates R's copies)
    log_w_gen0 = fwd_log_w[0]
    x_init = torch.zeros(batch_size, N, dim, dtype=dtype, device=device)
    log_tau_init = log_tau(flat(x_init), tvec(0.0)).reshape(batch_size, N, 1)

    V0 = fwd_log_v[0]  # value at children of initial particles
    O0 = V0 - log_tau_init

    # Per-child return and product, exactly as in the loop.  The initial
    # particles were not produced by any resampling, so they are all distinct
    # (each is its own group): there are no copies to average over and the
    # multi-step term is simply the per-child product (childless -> one-step).
    log_mean_R0, has_desc0 = _log_mean_exp_by_ancestor(R, ix_next0)
    M0 = torch.where(has_desc0 > 0, log_w_gen0 + log_mean_R0, O0)

    R0 = _log_td_blend(O0, M0, lam)
    target0 = (R0 + log_tau_init).reshape(BN)

    all_x_list.append(flat(x_init))
    all_t_list.append(torch.full((BN,), 0.0, dtype=dtype, device=device))
    all_tgt_list.append(target0)

    # Reverse to chronological order
    all_x_list = all_x_list[::-1]
    all_t_list = all_t_list[::-1]
    all_tgt_list = all_tgt_list[::-1]

    return (
        torch.cat(all_x_list, dim=0),
        torch.cat(all_t_list, dim=0),
        torch.cat(all_tgt_list, dim=0),
    )


# --------------------------------------------------------------------------- #
# Output type                                                                  #
# --------------------------------------------------------------------------- #


class FBRRTSamples(NamedTuple):
    """
    x       : [N, M, d]  particle positions at t_0, ..., t_{N-1}
    t       : [N]        time grid values (excludes t=1)
    v_hat   : [N, M]     one-step BSDE targets for v_theta
    weights : [N, M]     local-entropy LSMC regression weights (mean 1), used to
                         weight the least-squares fit -- NOT the bootstrap target.
                         See `_entropy_regression_weights` and Hawkins et al.
                         (2020, arXiv:2006.12444) Alg. 2 line 20 / eq. 23.
    """

    x: Tensor  # [N, M, d]
    t: Tensor  # [N]
    v_hat: Tensor  # [N, M]
    weights: Tensor  # [N, M]


def _entropy_regression_weights(values: Tensor, entropy_lambda: float) -> Tensor:
    """Local-entropy LSMC regression weights (Hawkins et al. 2020, eq. 21/23).

    Returns per-particle weights theta_j proportional to exp(values_j /
    entropy_lambda), normalised to unit mean so the overall loss scale is
    preserved.  In the paper these weights multiply the backward least-squares
    objective (concentrating value-function accuracy near high-value paths);
    they MUST NOT be folded into the bootstrap target (the target's child
    expectation is taken under the sampling measure P, eq. 22, i.e. unweighted).

    ``entropy_lambda = inf`` -> uniform weights (the unweighted LSMC of eq. 22).

    NB: the paper's heuristic rho (eq. 30) also adds the accumulated running
    cost along the path; here we use only the child value, matching the
    historical implementation.  This concentrates by instantaneous value rather
    than path value -- a deliberate simplification, not a correctness bug.
    """
    if entropy_lambda == float("inf"):
        return torch.ones_like(values)
    z = values / entropy_lambda
    z = z - z.max()
    w = z.exp()
    return w / w.mean().clamp_min(1e-30)


def _resample_fbrrt(weights: Tensor, n: int, method: str = "systematic") -> Tensor:
    """Return n indices sampled proportional to weights."""
    if method == "multinomial":
        return torch.multinomial(weights, n, replacement=True)
    elif method == "systematic":
        cumw = weights.cumsum(dim=0)
        u0 = torch.rand(1, device=weights.device, dtype=weights.dtype) / n
        us = u0 + torch.arange(n, device=weights.device, dtype=weights.dtype) / n
        return torch.searchsorted(cumw, us).clamp(0, len(weights) - 1)
    else:
        raise ValueError(f"Unknown resample_method: {method!r}")


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
    entropy_lambda  Temperature for the local-entropy weighting (Hawkins et al.
                    2020, eq. 21).  Used in TWO places: (i) the forward
                    branch-resampling proposal (exp(v/entropy_lambda)) that
                    concentrates particle coverage, and (ii) the returned
                    `weights`, which weight the backward least-squares
                    regression -- NOT the bootstrap target, which is always the
                    unweighted child mean.  inf -> uniform (unweighted LSMC).
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

        # One-step BSDE target (Hawkins et al. 2020, Alg. 2 line 14):
        #   y_hat = E_P[V(child) | parent] + (l^mu + z^T D) dt.
        # E_P[V(child)|parent] is a UNIFORM mean over the parent's B children
        # (they are i.i.d. under the sampling drift K, eq. 22).  It must NOT be
        # entropy-weighted: the local-entropy weights belong on the regression
        # loss (all_w below / eq. 23), not on the target.
        ev_next = v_ch_mb.mean(dim=1)  # [M]

        driver = driver_coeff * (grad_x_v**2).sum(dim=-1) * dt  # [M]

        all_x.append(parent_x)
        all_t.append(torch.full((M,), t_i, device=device, dtype=dtype))
        all_v_hat.append((ev_next + driver).detach())
        all_w.append(_entropy_regression_weights(ev_next, entropy_lambda))

    # add reward targets at t=1:
    all_x.append(x)
    all_t.append(torch.full((M,), 1, device=device, dtype=dtype))
    all_v_hat.append(reward(x))
    all_w.append(_entropy_regression_weights(reward(x), entropy_lambda))

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
    resample_method: str = "multinomial",
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
                "indices": indices,  # [M] -> [0, M*B): which child each survivor is
            }
        )

    # ------------------------------------------------------------------ #
    # Backward pass: GAE-style lambda return                               #
    # ------------------------------------------------------------------ #
    # At each step i we have:
    #   EV_{i+1}  = UNIFORM mean of v_theta over the parent's B children   [M]
    #   delta_i   = a*(1 - 2*alpha) * |grad_x_v|^2 * dt                     [M]
    #
    # Recursion (sweep from i=N-1 down to i=0):
    #   G_N = r(X_1)  (terminal: x is the final particle positions)
    #   G_i = delta_i + EV_{i+1} + lam * (Gnext_i - EV_{i+1})
    #
    # Ancestry alignment (fix for the resampling/GAE mismatch): the multi-step
    # return G lives on the *resampled* survivors x_{i+1} = children[indices_i].
    # Survivor m descends from parent indices_i[m] // B, so survivor m is NOT in
    # general the successor of parent m.  We therefore aggregate G back to each
    # parent by its true descendants (scatter-mean over origin = indices // B),
    # giving Gnext_i[p] = mean of downstream returns over parent p's surviving
    # children.  Parents with no surviving descendant fall back to the one-step
    # value EV (so the lambda term vanishes).  This mirrors the ancestor-indexed
    # aggregation used by the ancestral_* methods.
    #
    # Resampling MUST be multinomial here (hence the changed default above):
    # systematic resampling returns a near-diagonal index map
    # (indices[m] ~ m*B + shared_offset), so every parent's "surviving child" is
    # the SAME branch -- the descendants are not i.i.d. and the multi-step
    # return picks up a bias that GROWS with n_steps.  Multinomial resampling
    # gives each parent an independent descendant and the estimator is unbiased
    # across lambda (verified numerically at entropy_lambda=inf).
    #
    # NB: survivors are resampled proportional to exp(v/entropy_lambda), so for
    # FINITE entropy_lambda the scatter-mean over survivors is a value-biased
    # estimate of E_P[downstream | parent's children].  It is unbiased only as
    # entropy_lambda -> inf (uniform resampling); for the multi-step return,
    # prefer entropy_lambda=inf, or use ancestral_mc_td_lambda (log-space,
    # ancestor-exact).

    # Terminal: reward at final particle positions
    t_terminal = torch.full((M,), 1.0, device=device, dtype=dtype)
    G = reward(x).detach()  # [M], on the final-step survivors
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
        indices = data["indices"]  # [M] survivor -> child index

        # One-step term: UNIFORM mean over the parent's own B children (eq. 22).
        EV_next = v_ch_mb.mean(dim=1)  # [M]

        # Aggregate the downstream return G (on this step's survivors) back to
        # each parent by ancestry, so it pairs with the SAME parent as EV_next.
        origin = (indices // B).long()  # [M] parent index of each survivor
        sums = torch.zeros(M, device=device, dtype=dtype)
        sums.scatter_add_(0, origin, G)
        counts = torch.zeros(M, device=device, dtype=dtype)
        counts.scatter_add_(0, origin, torch.ones_like(G))
        Gnext = torch.where(counts > 0, sums / counts.clamp_min(1.0), EV_next)  # [M]

        # BSDE driver: delta_i = a*(1 - 2*alpha) * |grad_x_v|^2 * dt
        delta = driver_coeff * (grad_x_v**2).sum(dim=-1) * dt  # [M]

        # GAE recursion (now ancestry-aligned):
        # G_i = delta_i + EV_{i+1} + lam * (Gnext_i - EV_{i+1})
        G = delta + EV_next + lam * (Gnext - EV_next)  # [M], on parent_x (t_i)

        all_x.append(parent_x)
        all_t.append(torch.full((M,), t_i, device=device, dtype=dtype))
        all_v_hat.append(G.detach())
        all_w.append(_entropy_regression_weights(EV_next, entropy_lambda))

    # Reverse to chronological order
    all_x.reverse()
    all_t.reverse()
    all_v_hat.reverse()
    all_w.reverse()

    # Append terminal reward targets
    all_x.append(x)
    all_t.append(t_terminal)
    all_v_hat.append(reward(x).detach())
    all_w.append(_entropy_regression_weights(reward(x), entropy_lambda))

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

    Not part of Hawkins et al. (2020) -- that algorithm estimates Z analytically
    as ``sigma^T grad_x V`` (see ``fbrrt_smc_grad_control``).  This variant
    estimates ``grad_x V_target`` by anchoring on ``grad_x v_policy`` (autograd)
    and adding a Malliavin/IBP residual correction for ``v_target - v_policy``.

    .. note::

        The deterministic driver/scaling bugs of an earlier version are fixed:
        the driver now matches the analytic form
        ``a*(|z|^2 - 2*alpha * z.grad v_policy)*dt`` and the residual
        z-correction carries the required ``1/sqrt(2a)`` Malliavin factor, so the
        targets are unbiased when ``v_target == v_policy`` and approximately so
        while the two stay close.  A residual remains: the ``|z|^2`` term still
        contains the VARIANCE of the (finite-branch) Malliavin estimate of the
        residual gradient, which grows with ``|v_target - v_policy|`` and with
        ``1/branch``.  It is the same family as the instability that makes
        ``fbrrt_smc_grad_mc_Z`` diverge, but here it is anchored/variance-reduced
        by the control variate and is small in the intended regime where
        ``v_policy`` is a lagged/EMA copy of ``v_target``.  Use a large ``branch``
        and keep ``v_policy`` close to ``v_target`` for low bias.

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
    # We work with z := Z / sqrt(2a) = grad_x V (the un-scaled gradient).  The
    # RCV estimates grad_x V_target by anchoring on grad_x v_policy and adding a
    # Malliavin/integration-by-parts correction for the residual
    #     eps_b = v_target(x_{i+1}^b) - v_policy(x_{i+1}^b):
    #
    #     z_rcv = grad_x v_policy + grad_x(eps),
    #     grad_x(eps) ~= (1 / (sqrt(2a) * dt)) * sum_b w_b * eps_b * dW_b
    #
    # The 1/sqrt(2a) is REQUIRED: for X' = x + K dt + sqrt(2a) dW the IBP
    # estimator of grad_x E[phi] is E[phi dW] / (sqrt(2a) dt), not E[phi dW]/dt.
    # (Previously the sqrt(2a) was missing -- bias for v_target != v_policy.)
    #
    # The BSDE driver (added to E_P[V_target(child)|parent]) is the same one used
    # by fbrrt_smc_grad_control / _mc_Z, written with the RCV estimate of grad V:
    #
    #     driver = a * (|z_rcv|^2 - 2*alpha * z_rcv . grad_x v_policy) * dt
    #
    # It reduces to a*(1 - 2*alpha)*|grad v|^2 dt when eps -> 0 (z_rcv ->
    # grad v_policy).  (The earlier -|z|^2 + 2(1-alpha) z.grad form only matched
    # at eps = 0 and was biased once the control variate was active.)
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

            # Malliavin/IBP correction to z = grad_x V:
            #   grad_x(eps) ~= (1 / (sqrt(2a) * dt)) * sum_b w_b * eps_b * dW_b
            # [M, B, 1] * [M, B, d] -> sum over B -> [M, d]
            z_correction = (w_norm.unsqueeze(-1) * eps.unsqueeze(-1) * dW).sum(
                dim=1
            ) / (sq2a * dt)  # [M, d]

        # Full z_rcv = grad_x v_policy + residual correction  [M, d]
        # grad_x_v_policy is already detached (no graph); z_correction
        # is also no_grad, so z_rcv carries no gradient -- targets are
        # stop-gradient by construction.
        z_rcv = grad_x_v_policy + z_correction  # [M, d]

        # Driver using the RCV estimate of grad_x V (matches grad_control / _mc_Z):
        #   a * (|z_rcv|^2 - 2*alpha * z_rcv . grad_x_v_policy) * dt
        z_sq = (z_rcv**2).sum(dim=-1)  # [M]
        z_dot = (z_rcv * grad_x_v_policy).sum(dim=-1)  # [M]
        driver = a * (z_sq - 2.0 * alpha * z_dot) * dt  # [M]

        # Regression target: E_P[V_target(child)|parent] + driver.
        # UNIFORM mean over the parent's B children (eq. 22), NOT entropy-weighted.
        ev_next = v_ch_target.mean(dim=1)  # [M]

        all_x.append(parent_x)
        all_t.append(torch.full((M,), t_i, device=device, dtype=dtype))
        all_v_hat.append((ev_next + driver).detach())
        all_w.append(_entropy_regression_weights(ev_next, entropy_lambda))

    # Terminal condition  (x is the resampled final population)
    all_x.append(x)
    all_t.append(torch.full((M,), 1.0, device=device, dtype=dtype))
    all_v_hat.append(reward(x).detach())
    all_w.append(_entropy_regression_weights(reward(x), entropy_lambda))

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
    r"""
    FBRRT-SMC with MC estimate of Z and separated policy / target value
    functions.

    .. warning::

        NUMERICALLY UNSTABLE -- NOT RECOMMENDED.  This routine estimates the
        BSDE Z process by a raw Monte-Carlo / Malliavin formula
        ``Z = (1/dt) * mean_b[ Y_{i+1}^b * dW_b ]`` and then feeds ``Z**2`` into
        the driver.  A finite-branch MC estimate of Z has variance ~ 1/(B*dt),
        so ``(1/2)|Z|^2 * dt`` retains an O(1/B) per-step bias that ACCUMULATES
        over steps and blows up as dt -> 0.  Empirically the targets diverge to
        1e20+ / NaN once n_steps is moderately large (verified at n_steps>=10).
        Unlike Hawkins et al. (2020), which estimates Z analytically as
        ``sigma^T grad_x V`` (autograd), this MC-Z variant is a later
        experiment that is not in the paper.  Prefer ``fbrrt_smc_grad_control``
        (autograd Z) or ``fbrrt_smc_grad_control_variate`` (autograd anchor +
        variance-reduced residual), which do not square a high-variance Z.
        Left in place for reference / reproducibility only.

    Two value functions are accepted:

      v_policy  -- defines the SOC control u*(x,t) = grad_x v_policy.
                   Used to compute the sampling drift K and the Girsanov
                   correction D_t.  Freeze this (e.g. a lagged / EMA copy)
                   to stabilise exploration while v_target is being trained.

    The Z estimator is the residual control variate

        Z = 1/dt * mean[ Y_{i+1} * B_{t_i,t_{i+1}}]

    where Y_{i+1}, B_{t_i,t_{i+1}} range over the children.

    The control is 2a* alpha* grad v_policy:

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
