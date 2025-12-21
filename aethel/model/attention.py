from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class SlidingWindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 256):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, seq_len, _ = x.shape
        q = self.q_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        device = x.device
        positions = torch.arange(seq_len, device=device)
        dist = positions[None, :] - positions[:, None]
        local_mask = dist.abs() <= self.window_size
        scores = scores.masked_fill(~local_mask, float('-inf'))

        if mask is not None:
            if mask.dim() == 2:
                attn_mask = mask[:, None, None, :]
            else:
                attn_mask = mask
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        all_inf = torch.isinf(scores).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(all_inf, 0.0)

        attn = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn, v)
        out = attended.transpose(1, 2).contiguous().view(b, seq_len, self.dim)
        gated = self.gate * out
        return self.out_proj(gated)


__all__ = ["SlidingWindowAttention"]
