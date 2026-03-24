"""Encoder-only transformer for discrete sequence diffusion models."""

import torch
import torch.nn as nn

from .normalization import AdaLN
from .time_embedding import TimeEmbedding


class TransformerBlock(nn.Module):
    """Encoder transformer block with time conditioning via AdaLN.

    Architecture: AdaLN -> MHA -> Residual -> AdaLN -> FFN -> Residual

    Bidirectional self-attention (no causal mask), suitable for encoder-only
    and discrete diffusion use cases.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_heads: int = 8,
        hidden_mult: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        """Initialize TransformerBlock.

        Args:
            dim: Model dimension.
            cond_dim: Dimension of time conditioning embedding.
            num_heads: Number of attention heads.
            hidden_mult: FFN hidden dimension multiplier.
            dropout: Dropout in FFN layers.
            attention_dropout: Dropout applied to attention weights.
        """
        super().__init__()
        self.norm1 = AdaLN(dim, cond_dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm2 = AdaLN(dim, cond_dim)
        hidden_dim = int(dim * hidden_mult)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.ffn[0].weight)
        nn.init.zeros_(self.ffn[0].bias)
        # Small output init for stable residual learning
        nn.init.xavier_uniform_(self.ffn[-1].weight, gain=0.1)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, dim).
            cond: Time conditioning of shape (batch, cond_dim).
            key_padding_mask: Optional bool mask of shape (batch, seq_len).
                True at positions to ignore (e.g. padding tokens).

        Returns:
            Output of shape (batch, seq_len, dim).
        """
        # Self-attention with pre-norm
        h = self.norm1(x, cond)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + h

        # FFN with pre-norm
        h = self.norm2(x, cond)
        h = self.ffn(h)
        x = x + h

        return x


class DiscreteSequenceTransformer(nn.Module):
    """Encoder-only transformer for discrete diffusion on token sequences.

    Takes a sequence of integer tokens and a diffusion timestep, outputs
    a sequence of vectors of the same length. For discrete diffusion, set
    output_dim=vocab_size to obtain per-position logits over the vocabulary.

    Inputs tokens (integers) rather than one-hots or pre-computed embeddings:
    an embedding lookup is more memory-efficient and is equivalent in
    expressiveness to a linear projection of a one-hot.

    Architecture:
        1. Token embedding + learned positional embedding
        2. N bidirectional transformer blocks with AdaLN time conditioning
        3. Final AdaLN + linear output projection
    """

    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        max_seq_len: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        time_embed_dim: int | None = None,
        hidden_mult: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        """Initialize DiscreteSequenceTransformer.

        Args:
            vocab_size: Number of discrete token types.
            output_dim: Output vector dimension per position.
                Set to vocab_size for logits (typical for discrete diffusion).
            max_seq_len: Maximum supported sequence length.
            hidden_dim: Internal transformer dimension.
            num_layers: Number of transformer blocks.
            num_heads: Number of attention heads. Must divide hidden_dim.
            time_embed_dim: Time conditioning dimension. Defaults to hidden_dim.
            hidden_mult: FFN hidden dimension multiplier.
            dropout: Dropout in FFN layers.
            attention_dropout: Dropout applied to attention weights.
        """
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        time_embed_dim = time_embed_dim or hidden_dim

        # Input embeddings: token + learned position
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        # Time conditioning
        self.time_embed = TimeEmbedding(time_embed_dim, time_embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=hidden_dim,
                    cond_dim=time_embed_dim,
                    num_heads=num_heads,
                    hidden_mult=hidden_mult,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection with final AdaLN (consistent with ResNetMLP)
        self.final_norm = AdaLN(hidden_dim, time_embed_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: Integer token tensor of shape (batch, seq_len).
                Values must be in [0, vocab_size).
            t: Diffusion time tensor of shape (batch,) with values in [0, 1].
            key_padding_mask: Optional bool mask of shape (batch, seq_len).
                True at positions to ignore (e.g. padding tokens).

        Returns:
            Output tensor of shape (batch, seq_len, output_dim).
        """
        _batch, seq_len = tokens.shape

        # Token + positional embeddings
        positions = torch.arange(seq_len, device=tokens.device)
        x = self.token_embed(tokens) + self.pos_embed(positions)

        # Time conditioning vector
        cond = self.time_embed(t.flatten())

        # Bidirectional transformer blocks
        for block in self.blocks:
            x = block(x, cond, key_padding_mask=key_padding_mask)

        # Final norm + readout
        x = self.final_norm(x, cond)
        return self.output_proj(x)


class TokenSequenceScoreTransformer(DiscreteSequenceTransformer):
    """Discrete score network: output_dim + 1 = vocab_size, with the last token
    being the mask.

    Convenience subclass where the output dimension matches the vocabulary
    size, giving unnormalised log-probabilities (logits) over tokens at
    each sequence position. This is the standard output shape for discrete
    diffusion denoising networks.
    """

    def _init_weights(self) -> None:
        super()._init_weights()
        # The output should map to uniform logits initially.
        nn.init.zeros_(self.output_proj.weight)

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        **kwargs,
    ):
        """Initialize TokenSequenceScoreTransformer.

        Args:
            vocab_size: Number of discrete token types (also the output dim).
            max_seq_len: Maximum supported sequence length.
            hidden_dim: Internal transformer dimension.
            num_layers: Number of transformer blocks.
            num_heads: Number of attention heads.
            **kwargs: Forwarded to DiscreteSequenceTransformer.
        """
        super().__init__(
            vocab_size=vocab_size + 1,  # Add mask token
            output_dim=vocab_size,
            max_seq_len=max_seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            **kwargs,
        )
