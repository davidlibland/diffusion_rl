"""Tests for checkpointing in train_diff_llm."""

import torch
from torch.utils.data import DataLoader, Dataset

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from diffusion_rl.models.discrete_diffusion import DiffusionLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTokenDataset(Dataset):
    """Random token sequences that mimic a real dataset (returns raw tensors)."""

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int = 64):
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class _FakeDataModule(L.LightningDataModule):
    def __init__(self, vocab_size: int, seq_len: int, batch_size: int = 4):
        super().__init__()
        self._vocab_size = vocab_size
        self._seq_len = seq_len
        self._batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            _FakeTokenDataset(self._vocab_size, self._seq_len),
            batch_size=self._batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            _FakeTokenDataset(self._vocab_size, self._seq_len, n_samples=16),
            batch_size=self._batch_size,
        )


def _small_model(vocab_size: int = 8, seq_len: int = 16) -> DiffusionLLM:
    return DiffusionLLM(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        num_inference_steps=2,
    )


def _quiet_trainer(**kwargs) -> L.Trainer:
    return L.Trainer(
        enable_progress_bar=False,
        logger=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_checkpoint_last_ckpt_is_created(tmp_path):
    """ModelCheckpoint creates last.ckpt after training completes."""
    vocab_size, seq_len = 8, 16
    ckpt_dir = tmp_path / "ckpts"

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="diffusion_llm-{step}",
        save_last=True,
        every_n_train_steps=2,
    )
    trainer = _quiet_trainer(
        max_steps=4,
        val_check_interval=4,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1,
        callbacks=[checkpoint_cb],
    )
    trainer.fit(_small_model(vocab_size, seq_len), _FakeDataModule(vocab_size, seq_len))

    assert (ckpt_dir / "last.ckpt").exists(), "last.ckpt should be written after training"


def test_checkpoint_preserves_weights(tmp_path):
    """Weights loaded from checkpoint match those of the original trained model."""
    vocab_size, seq_len = 8, 16
    ckpt_dir = tmp_path / "ckpts"
    model = _small_model(vocab_size, seq_len)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="diffusion_llm-{step}",
        save_last=True,
        every_n_train_steps=2,
    )
    trainer = _quiet_trainer(
        max_steps=4,
        val_check_interval=4,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1,
        callbacks=[checkpoint_cb],
    )
    trainer.fit(model, _FakeDataModule(vocab_size, seq_len))

    last_ckpt = ckpt_dir / "last.ckpt"
    loaded = DiffusionLLM.load_from_checkpoint(str(last_ckpt))

    original_state = {k: v.cpu() for k, v in model.state_dict().items()}
    loaded_state = {k: v.cpu() for k, v in loaded.state_dict().items()}
    for key in original_state:
        assert torch.allclose(original_state[key], loaded_state[key]), (
            f"Weight mismatch for parameter '{key}' after loading checkpoint"
        )


def test_resume_continues_global_step(tmp_path):
    """Resuming from a checkpoint continues the global step count, not resets it."""
    vocab_size, seq_len = 8, 16
    ckpt_dir = tmp_path / "ckpts"

    # --- First run: train for 4 steps ---
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="diffusion_llm-{step}",
        save_last=True,
        every_n_train_steps=2,
    )
    trainer1 = _quiet_trainer(
        max_steps=4,
        val_check_interval=4,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1,
        callbacks=[checkpoint_cb],
    )
    trainer1.fit(_small_model(vocab_size, seq_len), _FakeDataModule(vocab_size, seq_len))
    assert trainer1.global_step == 4

    # --- Second run: resume and train to step 7 ---
    last_ckpt = str(ckpt_dir / "last.ckpt")
    trainer2 = _quiet_trainer(
        max_steps=7,
        val_check_interval=7,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1,
    )
    trainer2.fit(
        _small_model(vocab_size, seq_len),
        _FakeDataModule(vocab_size, seq_len),
        ckpt_path=last_ckpt,
    )

    assert trainer2.global_step == 7, (
        f"Expected global_step=7 after resuming from step 4, got {trainer2.global_step}"
    )


def test_resume_updates_weights(tmp_path):
    """Training steps after resuming from a checkpoint actually change the weights."""
    vocab_size, seq_len = 8, 16
    ckpt_dir = tmp_path / "ckpts"

    # --- First run: train 4 steps, save checkpoint ---
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="diffusion_llm-{step}",
        save_last=True,
        every_n_train_steps=4,
    )
    trainer1 = _quiet_trainer(
        max_steps=4,
        val_check_interval=4,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1,
        callbacks=[checkpoint_cb],
    )
    trainer1.fit(_small_model(vocab_size, seq_len), _FakeDataModule(vocab_size, seq_len))

    last_ckpt = str(ckpt_dir / "last.ckpt")
    weights_at_ckpt = {
        k: v.cpu().clone()
        for k, v in DiffusionLLM.load_from_checkpoint(last_ckpt).state_dict().items()
    }

    # --- Second run: resume and train 3 more steps ---
    model2 = _small_model(vocab_size, seq_len)
    trainer2 = _quiet_trainer(
        max_steps=7,
        val_check_interval=7,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1,
    )
    trainer2.fit(model2, _FakeDataModule(vocab_size, seq_len), ckpt_path=last_ckpt)

    weights_after_resume = {k: v.cpu() for k, v in model2.state_dict().items()}

    changed = [
        key
        for key in weights_at_ckpt
        if not torch.allclose(weights_at_ckpt[key], weights_after_resume[key])
    ]
    assert changed, "No parameters changed after resuming training — gradient updates are not being applied"
