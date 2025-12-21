from typing import Any, List, Optional

import torch
from torch import nn
from torch.nn import functional as F


class DeltaNetLayer(nn.Module):
    def __init__(self, dim: int, conv_kernel: int = 3):
        super().__init__()
        self.dim = dim
        self.norm = nn.RMSNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=1)
        self.ff = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, state: Optional[Any] = None) -> torch.Tensor:
        x_norm = self.norm(x)
        conv_in = x_norm.transpose(1, 2)
        conv_out = self.conv(conv_in).transpose(1, 2)
        delta = torch.tanh(conv_out + self.ff(x_norm))
        g = torch.sigmoid(self.gate(x_norm))
        return x + g * delta


class DeltaNetStack(nn.Module):
    def __init__(self, num_layers: int, dim: int, conv_kernel: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([DeltaNetLayer(dim=dim, conv_kernel=conv_kernel) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, state: Optional[Any] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, state=state)
        return x


__all__ = ["DeltaNetLayer", "DeltaNetStack"]
