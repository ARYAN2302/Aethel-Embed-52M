from typing import Optional

import torch
from torch import nn


class TitansLiteMemory(nn.Module):
    def __init__(self, dim: int = 1024, momentum: float = 0.9, decay: float = 0.01, surprise_scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.decay = decay
        self.surprise_scale = surprise_scale
        self.register_buffer("memory", torch.zeros(1, dim), persistent=False)

    def reset(self, batch_size: Optional[int] = None, device: Optional[torch.device] = None):
        size = batch_size or self.memory.size(0)
        dev = device or self.memory.device
        self.memory = torch.zeros(size, self.dim, device=dev)

    def update(
        self,
        hidden: torch.Tensor,
        prediction_error: Optional[torch.Tensor] = None,
        log_stats: bool = False,
        memory_override: Optional[torch.Tensor] = None,
        inplace: bool = True,
    ) -> torch.Tensor:
        """Update memory. If memory_override is provided, returns updated state without mutating internal buffer."""
        # hidden: (batch, seq, dim) or (batch, dim)
        if hidden.dim() == 3:
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden

        if prediction_error is None:
            surprise = pooled.norm(dim=-1, keepdim=True)
        else:
            surprise = prediction_error.norm(dim=-1, keepdim=True)
        gate = torch.sigmoid(self.surprise_scale * surprise)

        mem = memory_override if memory_override is not None else self.memory
        if mem.size(0) != pooled.size(0):
            mem = torch.zeros(pooled.size(0), self.dim, device=pooled.device)

        with torch.no_grad():
            decayed = mem * (1.0 - self.decay)
            updated = decayed * self.momentum + (1.0 - self.momentum) * pooled
            new_mem = gate * updated + (1 - gate) * decayed
            if inplace:
                self.memory = new_mem
        return new_mem.detach()

    def read(self) -> torch.Tensor:
        return self.memory


__all__ = ["TitansLiteMemory"]
