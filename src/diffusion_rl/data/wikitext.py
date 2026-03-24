"""
WikiText-103 as a PyTorch Lightning DataModule
===============================================
Strategy: tokenize the entire corpus once into a flat int64 tensor, cache it
to disk, then slice fixed-length blocks during training.

Each sample is a pair (x, y) where:
    x = tokens[i : i + block_size]          # input
    y = tokens[i + 1 : i + block_size + 1]  # target (shifted by 1)

This implements the autoregressive factorisation p(x) = ∏_t p(x_t | x_{<t}).
The number of samples is ⌊(N - 1) / block_size⌋ where N = total token count,
so the dataset is a partition of the token stream into non-overlapping blocks
(overlapping windows are also possible — see the note at the bottom).

Tokenizer: tiktoken GPT-2 BPE (vocab size 50257).

Usage
-----
  dm = WikiText103DataModule(cache_dir="./cache", block_size=1024, batch_size=32)
  trainer = L.Trainer(...)
  trainer.fit(model, dm)
"""

from pathlib import Path
from typing import Optional

import lightning as L
import tiktoken
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def _load_and_tokenize_ascii(split: str, cache_dir: Path) -> torch.Tensor:
    """
    Load a WikiText-103 split from HuggingFace, concatenate all non-empty
    lines, tokenize as ascii, and return a flat int64 tensor.
    Results are cached to <cache_dir>_ascii/<split>.pt to avoid re-tokenizing.
    """
    cache_path = cache_dir / f"{split}.pt"
    if cache_path.exists():
        print(f"Loading cached tokens from {cache_path}")
        return torch.load(cache_path)

    print(f"Tokenizing split='{split}' ...")

    ds = load_dataset(
        "Salesforce/wikitext",
        name="wikitext-103-raw-v1",
        split=split,
    )

    # Concatenate non-empty lines; encode_ordinary omits special tokens
    all_ids: list[int] = []
    for sample in ds:
        text = sample["text"].strip()
        if text:
            all_ids.extend([t for t in text.encode("ascii", errors="ignore")])

    tokens = torch.tensor(all_ids, dtype=torch.int64)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tokens, cache_path)
    print(f"  {len(tokens):,} tokens saved to {cache_path}")
    return tokens


def _load_and_tokenize(split: str, cache_dir: Path) -> torch.Tensor:
    """
    Load a WikiText-103 split from HuggingFace, concatenate all non-empty
    lines, tokenize with GPT-2 BPE, and return a flat int64 tensor.
    Results are cached to <cache_dir>/<split>.pt to avoid re-tokenizing.
    """
    cache_path = cache_dir / f"{split}.pt"
    if cache_path.exists():
        print(f"Loading cached tokens from {cache_path}")
        return torch.load(cache_path)

    print(f"Tokenizing split='{split}' ...")
    enc = tiktoken.get_encoding("gpt2")

    ds = load_dataset(
        "Salesforce/wikitext",
        name="wikitext-103-raw-v1",
        split=split,
    )

    # Concatenate non-empty lines; encode_ordinary omits special tokens
    all_ids: list[int] = []
    for sample in ds:
        text = sample["text"].strip()
        if text:
            all_ids.extend(enc.encode_ordinary(text))

    tokens = torch.tensor(all_ids, dtype=torch.int64)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tokens, cache_path)
    print(f"  {len(tokens):,} tokens saved to {cache_path}")
    return tokens


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TokenBlockDataset(Dataset):
    """
    Wraps a flat token tensor as a sequence of non-overlapping (x) blocks.

    Given a token sequence T of length N, the i-th sample is:
        x_i = T[i*B     : i*B + B]
    where B = block_size. The total number of samples is ⌊N / B⌋.
    """

    def __init__(self, tokens: torch.Tensor, block_size: int) -> None:
        self.tokens = tokens
        self.block_size = block_size
        # (N - 1) ensures every x_i has a valid target y_i
        self.n_samples = (len(tokens) - 1) // block_size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        x = self.tokens[start : start + self.block_size]
        return x


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class WikiText103DataModule(L.LightningDataModule):
    def __init__(
        self,
        cache_dir: str = "./wikitext103_cache",
        block_size: int = 1024,
        batch_size: int = 32,
        num_workers: int = 4,
        ascii=False,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir if not ascii else f"{cache_dir}_ascii")
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ascii = ascii

        self.train_dataset: Optional[TokenBlockDataset] = None
        self.val_dataset: Optional[TokenBlockDataset] = None
        self.test_dataset: Optional[TokenBlockDataset] = None

    def prepare_data(self) -> None:
        # Called once on rank 0. Downloads + tokenizes all splits.
        for split in ("train", "validation", "test"):
            if self.ascii:
                _load_and_tokenize_ascii(split, self.cache_dir)
            else:
                _load_and_tokenize(split, self.cache_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        # Called on every rank after prepare_data.
        def load(split):
            return TokenBlockDataset(
                _load_and_tokenize_ascii(split, self.cache_dir)
                if self.ascii
                else _load_and_tokenize(split, self.cache_dir),
                self.block_size,
            )

        if stage in ("fit", None):
            self.train_dataset = load("train")
            self.val_dataset = load("validation")
        if stage in ("test", None):
            self.test_dataset = load("test")

    def _make_loader(self, dataset: TokenBlockDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dm = WikiText103DataModule(
        cache_dir="./wikitext103_cache", block_size=1024, batch_size=4
    )
    dm.prepare_data()
    dm.setup("fit")

    train_ds = dm.train_dataset
    print(f"Train samples : {len(train_ds):,}")
    print(f"Val samples   : {len(dm.val_dataset):,}")

    x = train_ds[0]
    print(f"x shape: {x.shape}, dtype: {x.dtype}")
    enc = tiktoken.get_encoding("gpt2")
    print(enc.decode(x.tolist()))

    loader = dm.train_dataloader()
    xb = next(iter(loader))
    print(f"Batch x: {xb.shape}")

# ---------------------------------------------------------------------------
# Note on overlapping windows
# ---------------------------------------------------------------------------
# The non-overlapping scheme above wastes no tokens but means each token
# appears in exactly one training context. For a dataset this small you may
# prefer overlapping windows (stride < block_size), which gives
# n_samples = N - block_size samples and exposes each token to more varied
# left-contexts — at the cost of higher redundancy and correlated batches.
#
# To do that, change __getitem__ to:
#   start = idx                              # stride-1 sliding window
#   x = self.tokens[start : start + block_size]
#   y = self.tokens[start + 1 : start + block_size + 1]
# and n_samples = len(tokens) - block_size.
