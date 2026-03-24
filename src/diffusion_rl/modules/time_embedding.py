"""Sinusoidal time embeddings for diffusion models."""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for time steps.

    Follows the standard formulation from "Attention Is All You Need"
    and adapted for continuous time in diffusion models.
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        """Initialize the time embedding.

        Args:
            dim: Embedding dimension (must be even).
            max_period: Maximum period for the sinusoidal functions.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dim must be even, got {dim}")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embeddings for time steps.

        Args:
            t: Time tensor of shape (batch_size,) with values in [0, 1].

        Returns:
            Embeddings of shape (batch_size, dim).
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimeEmbedding(nn.Module):
    """Time embedding with MLP projection.

    Combines sinusoidal embedding with a small MLP to produce
    richer time representations.
    """

    def __init__(self, time_dim: int, embed_dim: int):
        """Initialize the time embedding.

        Args:
            time_dim: Dimension of sinusoidal embedding.
            embed_dim: Output dimension after MLP projection.
        """
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize MLP weights with Xavier uniform."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute time embeddings.

        Args:
            t: Time tensor of shape (batch_size,) with values in [0, 1].

        Returns:
            Embeddings of shape (batch_size, embed_dim).
        """
        return self.mlp(self.sinusoidal(t))
