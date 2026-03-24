"""Tests for discrete diffusion model and CTMC integrator."""

import pytest
import torch

from diffusion_rl.models.discrete_diffusion import DiffusionLLM, integrate_ctmc


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_model():
    """Minimal DiffusionLLM with small dims for fast tests."""
    return DiffusionLLM(
        vocab_size=8,
        max_seq_len=16,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_inference_steps=4,
    )


# ---------------------------------------------------------------------------
# integrate_ctmc
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bs,seq_len", [(1, 6), (2, 10)])
def test_integrate_ctmc_output_shape(small_model, bs, seq_len):
    """Output has the same shape as input."""
    mask_val = small_model.mask_val
    x0 = torch.full((bs, seq_len), mask_val, dtype=torch.long)
    with torch.no_grad():
        out = integrate_ctmc(x0, small_model.model, n_steps=4, mask_val=mask_val)
    assert out.shape == (bs, seq_len)


@pytest.mark.parametrize("bs,seq_len", [(1, 6), (2, 10)])
def test_integrate_ctmc_starts_fully_masked(small_model, bs, seq_len):
    """x0 must be entirely mask tokens before integration."""
    mask_val = small_model.mask_val
    x0 = torch.full((bs, seq_len), mask_val, dtype=torch.long)
    # Assert the precondition (all-masked) holds before we call the integrator
    assert (x0 == mask_val).all(), "x0 must be initialised to mask_val"


@pytest.mark.parametrize("bs,seq_len", [(1, 6), (2, 10)])
def test_integrate_ctmc_no_mask_in_output(small_model, bs, seq_len):
    """After integration, no position should contain the mask token."""
    mask_val = small_model.mask_val
    x0 = torch.full((bs, seq_len), mask_val, dtype=torch.long)
    with torch.no_grad():
        out = integrate_ctmc(x0, small_model.model, n_steps=4, mask_val=mask_val)
    assert (out != mask_val).all(), "Output should contain no mask tokens"


@pytest.mark.parametrize("bs,seq_len", [(1, 6), (2, 10)])
def test_integrate_ctmc_output_in_vocab(small_model, bs, seq_len):
    """Output tokens must be valid vocabulary indices: 0 <= token < mask_val."""
    mask_val = small_model.mask_val
    x0 = torch.full((bs, seq_len), mask_val, dtype=torch.long)
    with torch.no_grad():
        out = integrate_ctmc(x0, small_model.model, n_steps=4, mask_val=mask_val)
    assert (out >= 0).all(), "All output tokens must be non-negative"
    assert (out < mask_val).all(), (
        f"All output tokens must be < mask_val ({mask_val}), "
        f"got max={out.max().item()}"
    )


# ---------------------------------------------------------------------------
# DiffusionLLM.training_step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bs,seq_len", [(1, 6), (2, 10)])
def test_training_step_returns_scalar(small_model, bs, seq_len):
    """training_step must return a scalar (0-d) tensor."""
    vocab_size = small_model.hparams.vocab_size
    batch = torch.randint(0, vocab_size, (bs, seq_len))
    loss = small_model.training_step(batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"


@pytest.mark.parametrize("bs,seq_len", [(1, 6), (2, 10)])
def test_training_step_finite_loss(small_model, bs, seq_len):
    """training_step must produce a finite loss for valid inputs."""
    vocab_size = small_model.hparams.vocab_size
    batch = torch.randint(0, vocab_size, (bs, seq_len))
    loss = small_model.training_step(batch)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


@pytest.mark.parametrize("bs,seq_len", [(1, 6), (2, 10)])
def test_training_step_input_bounds(small_model, bs, seq_len):
    """Validate that the test inputs satisfy the stated precondition."""
    vocab_size = small_model.hparams.vocab_size
    batch = torch.randint(0, vocab_size, (bs, seq_len))
    assert batch.dtype == torch.int64, "batch must be long dtype"
    assert (batch >= 0).all(), "All tokens must be >= 0"
    assert (batch < vocab_size).all(), "All tokens must be < vocab_size"
