from typing import Any, Dict

import torch
from transformers import AutoModel, AutoTokenizer


def _bag_of_words_sparse(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    # Simple SPLADE-lite style bag-of-words activation.
    bow = torch.zeros(input_ids.size(0), vocab_size, device=input_ids.device)
    bow.scatter_add_(1, input_ids, torch.ones_like(input_ids, dtype=bow.dtype))
    return torch.log1p(bow)


def load_teacher(model_name: str, device: str = "cpu", produce_sparse: bool = False) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def encode(input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Truncate to teacher max length to avoid position-mismatch errors.
        max_len = getattr(model.config, "max_position_embeddings", input_ids.size(1))
        if input_ids.size(1) > max_len:
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Masked mean pooling on last_hidden_state keeps teacher/student alignment.
            hidden = outputs.last_hidden_state
            mask_f = attention_mask.float()
            denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1e-6)
            pooled = (hidden * mask_f.unsqueeze(-1)).sum(dim=1) / denom
            if not torch.isfinite(pooled).all():
                pooled = torch.nan_to_num(pooled)
            out: Dict[str, Any] = {"dense": pooled}
            if produce_sparse:
                out["sparse_logits"] = _bag_of_words_sparse(input_ids, tokenizer.vocab_size)
            return out

    return {"model": model, "tokenizer": tokenizer, "encode": encode}


__all__ = ["load_teacher"]
