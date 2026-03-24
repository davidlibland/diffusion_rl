"""Adaptive Layer Normalization for time-conditioned networks."""

import torch
import torch.nn as nn


class AdaLN(nn.Module):
    """Adaptive Layer Normalization.

    Applies layer normalization where scale and shift parameters are
    computed from a conditioning signal (e.g., time embedding).
    This follows the approach used in DiT and other modern architectures.

    The output is: scale * LayerNorm(x) + shift
    where scale and shift are projected from the conditioning embedding.
    """

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        """Initialize AdaLN.

        Args:
            dim: Feature dimension to normalize.
            cond_dim: Dimension of conditioning embedding.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.proj = nn.Linear(cond_dim, 2 * dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection to output zeros.

        This makes AdaLN start as identity (scale=1, shift=0),
        which helps with training stability.
        """
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply adaptive layer normalization.

        Args:
            x: Input tensor of shape (batch_size, ..., dim).
            cond: Conditioning tensor of shape (batch_size, cond_dim).

        Returns:
            Normalized tensor of shape (batch_size, ..., dim).
        """
        # Project conditioning to scale and shift
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)

        # Reshape for broadcasting: (batch_size, 1, ..., 1, dim)
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

        # Apply: (1 + scale) * norm(x) + shift
        # Using (1 + scale) so that zero-initialized scale gives identity
        return (1 + scale) * self.norm(x) + shift
