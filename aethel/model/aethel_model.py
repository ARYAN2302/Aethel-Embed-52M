from typing import Any, Dict, Optional

import torch
from torch import nn

from .embeddings import TokenEmbedding
from .deltanet import DeltaNetStack
from .attention import SlidingWindowAttention
from .memory_titans_lite import TitansLiteMemory
from .hybrid_head import HybridHead


class AethelModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size=vocab_size, dim=dim)
        self.deltanet = DeltaNetStack(num_layers=6, dim=dim)
        self.attn = nn.ModuleList([SlidingWindowAttention(dim=dim, num_heads=num_heads) for _ in range(2)])
        self.memory = TitansLiteMemory(dim=dim)
        self.memory_proj = nn.Linear(dim * 2, dim)
        self.head = HybridHead(dense_dim=dim, sparse_vocab=5000)

        with torch.no_grad():
            self.embedding.token.weight.mul_(1.5)

    def pooled_mean(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        mask = mask.float()
        masked = x * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True)
        # If the sequence is fully padded, fall back to unmasked mean to avoid zero embeddings.
        if (denom == 0).any():
            return x.mean(dim=1)
        denom = denom.clamp(min=1e-6)
        return masked.sum(dim=1) / denom

    def forward(self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None, log_memory: bool = False) -> Dict[str, Any]:
        x = self.embedding(token_ids)
        x = self.deltanet(x)
        for attn in self.attn:
            x = attn(x, mask=mask)
        pooled = self.pooled_mean(x, mask=mask)
        # Include explicit memory state; project concat back to model dim.
        memory_state = self.memory.update(x, log_stats=log_memory, inplace=True)
        fused = torch.cat([pooled, memory_state], dim=-1)
        fused = self.memory_proj(fused)
        fused_norm = fused.norm(dim=-1, keepdim=True)
        if (fused_norm == 0).any():
            fused = fused + 1e-3 * torch.randn_like(fused)
        outputs = self.head(fused)
        outputs["memory"] = memory_state
        outputs["sequence_hidden"] = x
        return outputs

    def forward_with_memory(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        memory_state: Optional[torch.Tensor] = None,
        log_memory: bool = False,
    ) -> Dict[str, Any]:
        """Stateful forward: accepts external memory_state and returns next state without mutating internal buffer."""
        x = self.embedding(token_ids)
        x = self.deltanet(x)
        for attn in self.attn:
            x = attn(x, mask=mask)
        pooled = self.pooled_mean(x, mask=mask)
        next_memory = self.memory.update(x, log_stats=log_memory, memory_override=memory_state, inplace=False)
        fused = torch.cat([pooled, next_memory], dim=-1)
        fused = self.memory_proj(fused)
        fused_norm = fused.norm(dim=-1, keepdim=True)
        if (fused_norm == 0).any():
            fused = fused + 1e-3 * torch.randn_like(fused)
        outputs = self.head(fused)
        outputs["memory"] = next_memory
        outputs["sequence_hidden"] = x
        return outputs


__all__ = ["AethelModel"]
