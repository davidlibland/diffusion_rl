"""Tests asserting that OnPolicyValue and OffPolicyValue follow the same
training and inference logic when given the same data and value module.

The on-policy and off-policy LightningModules differ in two non-trivial ways:
  - OnPolicyValue keeps an EMA shadow (`self.ema`) and `drift` defaults to
    `use_ema=True`, while OffPolicyValue always uses the live `value_module`.
  - The training-step batch tuples differ: on-policy receives a precomputed
    target `y`, off-policy receives `x1` and computes `target = reward(x1)`
    inside the step.

These tests pin down that, conditioned on identical inputs (same value
module weights, same `(x, t)`, and same effective target), the two paths
produce bit-equal losses, gradients, and drift outputs.
"""

import pytest
import torch
import torch.nn as nn

from diffusion_rl.models.off_policy import OffPolicyValue
from diffusion_rl.models.on_policy import OnPolicyValue


# ─── Tiny components ──────────────────────────────────────────────────────


class TinyValue(nn.Module):
    """Small (x, t) → scalar network used as a stand-in for ValueNetwork."""

    def __init__(self, dim: int = 2, hidden: int = 16, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.lin1 = nn.Linear(dim + 1, hidden)
        self.lin2 = nn.Linear(hidden, 1)
        # deterministic init so independently-built copies match
        for p in self.parameters():
            p.data = torch.empty_like(p).normal_(generator=gen) * 0.3

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        h = torch.cat([x, t], dim=-1)
        return self.lin2(torch.tanh(self.lin1(h))).squeeze(-1)


def _no_op_log(*args, **kwargs):
    """Replace `self.log` so training_step can run without a Trainer."""
    return None


def _zero_drift(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def _quadratic_reward(x: torch.Tensor) -> torch.Tensor:
    return -0.5 * (x * x).sum(dim=-1)


# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def shared_value():
    """A single value module reused (by reference) by both modules."""
    torch.manual_seed(0)
    return TinyValue(dim=2, seed=42)


@pytest.fixture()
def batch():
    torch.manual_seed(1)
    bs, dim = 8, 2
    x1 = torch.randn(bs, dim)
    eps = torch.randn(bs, dim)
    t = torch.rand(bs, 1)
    a = 1.0
    x = t * x1 + torch.sqrt(2 * a * t * (1 - t)) * eps
    return x1, x, t


def _make_modules(value, loss_type: str = "quad", lr: float = 1e-3, dim: int = 2,
                  ema_decay: float = 0.99):
    """Build OnPolicyValue and OffPolicyValue sharing the same value module."""
    on = OnPolicyValue(
        base_score_module=_zero_drift,
        value_module=value,
        reward_function=_quadratic_reward,
        a=1.0, lr=lr, dim=dim,
        loss_type=loss_type, ema_decay=ema_decay,
    )
    off = OffPolicyValue(
        base_score_module=_zero_drift,
        value_module=value,           # same instance, shared parameters
        reward_function=_quadratic_reward,
        a=1.0, lr=lr, dim=dim,
        loss_type=loss_type,
    )
    on.log = _no_op_log
    off.log = _no_op_log
    return on, off


# ─── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("loss_type", ["quad", "mse"])
def test_training_step_loss_matches(shared_value, batch, loss_type):
    """Same `(x, t)` and same target → on-policy & off-policy compute identical losses."""
    on, off = _make_modules(shared_value, loss_type=loss_type)
    x1, x, t = batch

    target = _quadratic_reward(x1)        # what off-policy will compute internally
    loss_on = on.training_step((target, x, t), batch_idx=0)
    loss_off = off.training_step((x1, x, t), batch_idx=0)

    assert torch.allclose(loss_on, loss_off, rtol=0, atol=0), (
        f"loss mismatch: on={loss_on.item()}, off={loss_off.item()}"
    )


@pytest.mark.parametrize("loss_type", ["quad", "mse"])
def test_training_step_gradient_matches(batch, loss_type):
    """The two paths produce identical gradients on the value-module parameters.

    Uses two independent (but identically initialised) value modules so each
    LightningModule has its own `.grad` to inspect, then compares them
    parameter-by-parameter.
    """
    v_on = TinyValue(dim=2, seed=42)
    v_off = TinyValue(dim=2, seed=42)
    # sanity: identical init
    for p_on, p_off in zip(v_on.parameters(), v_off.parameters()):
        assert torch.equal(p_on.data, p_off.data)

    on, _ = _make_modules(v_on, loss_type=loss_type)
    _, off = _make_modules(v_off, loss_type=loss_type)
    x1, x, t = batch
    target = _quadratic_reward(x1)

    loss_on = on.training_step((target, x, t), batch_idx=0)
    loss_off = off.training_step((x1, x, t), batch_idx=0)
    loss_on.backward()
    loss_off.backward()

    for (n_on, p_on), (n_off, p_off) in zip(
        v_on.named_parameters(), v_off.named_parameters()
    ):
        assert n_on == n_off
        assert p_on.grad is not None and p_off.grad is not None, n_on
        assert torch.allclose(p_on.grad, p_off.grad, rtol=0, atol=0), (
            f"grad mismatch on {n_on}: max diff = "
            f"{(p_on.grad - p_off.grad).abs().max().item():.2e}"
        )


def test_drift_matches_with_use_ema_false(shared_value, batch):
    """`OnPolicyValue.drift(use_ema=False)` ≡ `OffPolicyValue.drift`."""
    on, off = _make_modules(shared_value)
    _, x, t = batch
    t_flat = t.squeeze(-1)

    drift_on = on.drift(x, t_flat, use_ema=False)
    drift_off = off.drift(x, t_flat)

    assert drift_on.shape == drift_off.shape == x.shape
    assert torch.allclose(drift_on, drift_off, rtol=0, atol=0), (
        f"drift mismatch (no EMA): max diff = "
        f"{(drift_on - drift_off).abs().max().item():.2e}"
    )


def test_drift_matches_at_init_even_with_ema(shared_value, batch):
    """At init, EMA = deepcopy(value), so on-policy(use_ema=True) ≡ off-policy."""
    on, off = _make_modules(shared_value)
    _, x, t = batch
    t_flat = t.squeeze(-1)

    drift_on_ema = on.drift(x, t_flat, use_ema=True)
    drift_off = off.drift(x, t_flat)

    assert torch.allclose(drift_on_ema, drift_off, rtol=0, atol=0), (
        "Before any EMA update, the shadow must equal the live network exactly."
    )


def test_drift_diverges_after_value_update(shared_value, batch):
    """After the live `value_module` updates (and EMA tracks with decay<1),
    the EMA-driven drift must differ from the live-driven drift.

    This is the inverse of the equivalence tests: it pins down that the
    EMA branch is *not* a no-op once training has started, which would
    otherwise make `use_ema` semantically irrelevant.
    """
    on, off = _make_modules(shared_value, ema_decay=0.5)  # fast EMA mixing
    _, x, t = batch
    t_flat = t.squeeze(-1)

    # Perturb the live parameters (simulate a gradient step); EMA shadow lags.
    with torch.no_grad():
        for p in shared_value.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    on.ema.update(shared_value)   # EMA shadow ≠ live now

    drift_live = on.drift(x, t_flat, use_ema=False)
    drift_ema = on.drift(x, t_flat, use_ema=True)
    drift_off = off.drift(x, t_flat)

    # Live and off-policy still match (same module, same code path).
    assert torch.allclose(drift_live, drift_off, rtol=0, atol=0)

    # EMA path diverges.
    diff = (drift_ema - drift_live).abs().max().item()
    assert diff > 1e-3, (
        f"EMA-driven drift unexpectedly equal to live-driven drift "
        f"(max diff = {diff:.2e}); EMA may be a no-op."
    )


def test_integrate_sde_matches_when_seeded(shared_value, batch):
    """Trajectory rollouts agree when the noise seed is matched and EMA is off.

    This is the "validation_step would compute the same val_reward" property.
    """
    from diffusion_rl.algorithms.integration import integrate_sde

    on, off = _make_modules(shared_value)
    _, x, _ = batch
    n = 4
    x0 = x[:n].clone()

    torch.manual_seed(7)
    traj_on = integrate_sde(
        x0, drift=lambda xx, tt: on.drift(xx, tt, use_ema=False),
        a=on.a, n_steps=10,
    )
    torch.manual_seed(7)
    traj_off = integrate_sde(x0, drift=off.drift, a=off.a, n_steps=10)

    assert torch.allclose(traj_on, traj_off, rtol=0, atol=0), (
        f"trajectory mismatch: max diff = "
        f"{(traj_on - traj_off).abs().max().item():.2e}"
    )
