"""Token embeddings with optional RoPE positional encoding."""

from typing import Optional

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Applies rotary position embeddings to q/k vectors."""

    def __init__(self, dim: int, max_position: int = 32768, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if position_ids is None:
            cos = self.cos_cached[: x.size(-2)]
            sin = self.sin_cached[: x.size(-2)]
        else:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return (x * cos) + (rotated * sin)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int, rope: bool = True, max_position: int = 32768):
        super().__init__()
        self.dim = dim
        self.token = nn.Embedding(vocab_size, dim)
        self.rope = RotaryEmbedding(dim=dim, max_position=max_position) if rope else None

    def forward(self, token_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.token(token_ids)
        if self.rope is not None:
            # Apply RoPE on last two dims (seq, embed)
            x = self.rope(x, position_ids=position_ids)
        return x


__all__ = ["TokenEmbedding", "RotaryEmbedding"]
