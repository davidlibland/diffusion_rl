"""ResNet-style MLP with time conditioning for diffusion models."""

import torch
import torch.nn as nn

from .normalization import AdaLN
from .time_embedding import TimeEmbedding


class ResNetBlock(nn.Module):
    """Pre-norm ResNet block with time conditioning.

    Architecture: AdaLN -> Linear -> Activation -> Linear -> Residual
    Uses pre-normalization for training stability.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        hidden_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        """Initialize ResNet block.

        Args:
            dim: Input and output dimension.
            cond_dim: Dimension of conditioning embedding.
            hidden_mult: Multiplier for hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        hidden_dim = int(dim * hidden_mult)
        self.norm = AdaLN(dim, cond_dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform, output layer with small values."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # Initialize output layer with small values for stable residual learning
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch_size, dim).
            cond: Conditioning tensor of shape (batch_size, cond_dim).

        Returns:
            Output tensor of shape (batch_size, dim).
        """
        h = self.norm(x, cond)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class ResNetMLP(nn.Module):
    """ResNet-style MLP with time conditioning.

    A simple but effective architecture for score networks in flow/diffusion
    models. Can also be used as a value network for RL by setting the output
    dimension appropriately.

    Architecture:
        1. Input projection: input_dim -> hidden_dim
        2. Stack of ResNet blocks with AdaLN conditioning
        3. Final normalization and output projection

    The time embedding is computed internally and used to condition
    all normalization layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        time_embed_dim: int | None = None,
        hidden_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        """Initialize ResNetMLP.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
            hidden_dim: Hidden dimension for residual blocks.
            num_blocks: Number of ResNet blocks.
            time_embed_dim: Time embedding dimension. Defaults to hidden_dim.
            hidden_mult: Hidden dimension multiplier within blocks.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        time_embed_dim = time_embed_dim or hidden_dim
        self.time_embed = TimeEmbedding(time_embed_dim, time_embed_dim)

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList(
            [
                ResNetBlock(hidden_dim, time_embed_dim, hidden_mult, dropout)
                for _ in range(num_blocks)
            ]
        )

        # Output projection with final normalization
        self.final_norm = AdaLN(hidden_dim, time_embed_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize input/output projection weights."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        # Small initialization for output to help with initial training
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            t: Time tensor of shape (batch_size,) with values in [0, 1].

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        # Compute time embedding
        cond = self.time_embed(t.flatten())

        # Input projection
        h = self.input_proj(x)

        # Residual blocks
        for block in self.blocks:
            h = block(h, cond)

        # Output projection
        h = self.final_norm(h, cond)
        return self.output_proj(h)


class ScoreNetwork(ResNetMLP):
    """Score network for flow/diffusion models.

    Convenience wrapper where output_dim defaults to input_dim,
    as the score has the same dimension as the data.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        **kwargs,
    ):
        """Initialize ScoreNetwork.

        Args:
            dim: Data dimension (both input and output).
            hidden_dim: Hidden dimension for residual blocks.
            num_blocks: Number of ResNet blocks.
            **kwargs: Additional arguments passed to ResNetMLP.
        """
        super().__init__(
            input_dim=dim,
            output_dim=dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            **kwargs,
        )


class ValueNetwork(ResNetMLP):
    """Value network for RL.

    Convenience wrapper with output_dim=1 for scalar value prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        bias=None,
        reward_fn=None,
        k=1,
        **kwargs,
    ):
        """Initialize ValueNetwork.

        Args:
            input_dim: State/observation dimension.
            hidden_dim: Hidden dimension for residual blocks.
            num_blocks: Number of ResNet blocks.
            **kwargs: Additional arguments passed to ResNetMLP.
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            **kwargs,
        )
        if bias is not None:
            self.output_proj.bias.data.fill_(bias)
        self.reward_fn = reward_fn
        self.k = k

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass returning scalar values.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            t: Time tensor of shape (batch_size,) with values in [0, 1].

        Returns:
            Value tensor of shape (batch_size,).
        """
        pred = super().forward(x, t).squeeze(-1)
        if self.reward_fn is not None:
            reward_at_1 = self.reward_fn(x)
            return reward_at_1 * t.flatten() ** self.k + pred * (
                1 - t.flatten() ** self.k
            )
        return pred
