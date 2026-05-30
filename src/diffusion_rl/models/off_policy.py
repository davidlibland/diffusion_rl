import lightning as L
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import IterableDataset

from diffusion_rl.algorithms.integration import integrate_sde
from diffusion_rl.losses.log_quadratic_bregman import log_quadratic_bregman_divergence
from typing import Callable


class InterpolatingNumpyDataset(IterableDataset):
    r"""
    Streams data from the generating function, producing data in batches

    Data is of the form x = t*x1+sqrt{2*a*t*(1-t)}*epsilon where:
     - epsilon is normally distributed
     - x1 is sampled from the generating function
     - t \in [0,1] is randomly sampled

    The generator yields batches of the form (x1, x, t)

    Args:
        generating_function: A function which takes a batch size bs and produces an array of shape (bs, d)
        a: The noise level in the interpolation
    """

    def __init__(
        self,
        generating_function: Callable[[int], np.ndarray],
        a: float = 1,
        batch_size: int = 1024,
    ):
        self.generating_function = generating_function
        self.batch_size = batch_size
        self.a = a
        self._x = None
        self._x1 = None
        self._t = None
        self._loc = 0

    def __iter__(self):
        # This generator function runs in an infinite loop
        while True:
            if self._x is None or self._loc >= self._x.shape[0]:
                # Generate a new batch:
                np_batch = self.generating_function(self.batch_size)
                self._x1 = torch.from_numpy(np_batch).to(dtype=torch.float)
                epsilon_batch = torch.randn_like(self._x1, dtype=torch.float)
                self._t = torch.rand(self.batch_size, 1, dtype=torch.float)
                # Compute x given x1, t, and epsilon
                self._x = (
                    self._t * self._x1
                    + torch.sqrt(2 * self.a * self._t * (1 - self._t)) * epsilon_batch
                )
                self._loc = 0

            x = self._x[self._loc]
            x1 = self._x1[self._loc]
            t = self._t[self._loc]
            yield x1, x, t
            self._loc += 1


# define the LightningModule
_T_BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
_T_BIN_NAMES = ["t00_20", "t20_40", "t40_60", "t60_80", "t80_100"]


class OffPolicyValue(L.LightningModule):
    def __init__(
        self,
        base_score_module,
        reward_function,
        value_module,
        a,
        lr,
        dim: int = 2,
        loss_type: str = "mse",
        grad_decay: float = None,
        analytical_value_fn=None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["base_score_module", "reward_function", "value_module", "analytical_value_fn"]
        )
        self.base_score_module = base_score_module
        self.reward_function = reward_function
        self.value_module = value_module
        self.loss_type = loss_type
        self.a = a
        self.analytical_value_fn = analytical_value_fn

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x1, x, t = batch
        if self.hparams.grad_decay is not None:
            x = x.clone().detach().requires_grad_(True)
        pred_value = self.value_module(x, t.flatten()).flatten()[:, None]
        true_value = self.reward_function(x1).flatten()[:, None]
        if self.loss_type == "mse":
            loss = nn.functional.mse_loss(torch.exp(pred_value), torch.exp(true_value))
        elif self.loss_type == "quad":
            loss = log_quadratic_bregman_divergence(pred_value, true_value).mean()
        self.log("train_loss", loss)
        # Per-bin variance of (r(x1) - V_analytical(x,t)), measuring off-policy target noise
        if self.analytical_value_fn is not None:
            with torch.no_grad():
                t_flat = t.flatten()
                v_anal = self.analytical_value_fn(x.detach(), t_flat)
                target_err = true_value.flatten() - v_anal
            for name, lo, hi in zip(_T_BIN_NAMES, _T_BIN_EDGES[:-1], _T_BIN_EDGES[1:]):
                mask = (t_flat >= lo) & (t_flat < hi)
                if mask.sum() > 1:
                    self.log(f"target_var_{name}", target_err[mask].var(),
                             on_step=True, on_epoch=False, prog_bar=False)

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
