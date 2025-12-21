from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F


class HybridHead(nn.Module):
    def __init__(self, dense_dim: int = 768, sparse_vocab: int = 5000, matryoshka_slices=(768, 512, 256, 128)):
        super().__init__()
        self.dense_dim = dense_dim
        self.sparse_vocab = sparse_vocab
        self.matryoshka_slices = tuple(sorted(matryoshka_slices, reverse=True))
        self.dense_proj = nn.Linear(dense_dim, dense_dim)
        self.sparse_proj = nn.Linear(dense_dim, sparse_vocab)

        with torch.no_grad():
            self.dense_proj.weight.mul_(1.2)
            self.sparse_proj.weight.mul_(1.2)

    def forward(self, pooled: torch.Tensor) -> Dict[str, Any]:
        dense_full = self.dense_proj(pooled)
        slices = {s: dense_full[..., :s] for s in self.matryoshka_slices if s <= dense_full.size(-1)}
        sparse_logits = self.sparse_proj(pooled)
        sparse_activations = torch.log1p(F.relu(sparse_logits))
        return {
            "dense": dense_full,
            "dense_slices": slices,
            "sparse_logits": sparse_logits,
            "sparse_activations": sparse_activations,
        }


__all__ = ["HybridHead"]
