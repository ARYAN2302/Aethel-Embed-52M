from typing import Any

import torch


def quantize_model(model: Any, method: str = "int8"):
    model.eval()
    if method == "int8":
        qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    elif method == "int4":
        # PyTorch lacks native int4; fall back to int8 as a placeholder.
        qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    else:
        raise ValueError(f"Unsupported quantization method: {method}")
    return qmodel


__all__ = ["quantize_model"]
