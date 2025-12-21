from typing import Any

import torch


def export_to_onnx(model: Any, path: str, seq_len: int = 256, opset: int = 17):
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randint(0, 1000, (1, seq_len), device=device)
    dummy_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)

    def _forward(input_ids, attention_mask):
        out = model(input_ids, mask=attention_mask)
        return out["dense"], out["sparse_logits"] if "sparse_logits" in out else out["dense"]

    torch.onnx.export(
        model,
        (dummy_input, dummy_mask),
        path,
        input_names=["input_ids", "attention_mask"],
        output_names=["dense", "sparse"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}},
        opset_version=opset,
    )
    print(f"Exported ONNX to {path}")


__all__ = ["export_to_onnx"]
