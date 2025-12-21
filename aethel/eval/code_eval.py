from typing import Any, Dict

import torch
from datasets import load_dataset
from itertools import islice
from transformers import AutoTokenizer


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


def run_code_eval(model: Any, tokenizer_name: str = "BAAI/bge-base-en-v1.5", split: str = "train", limit: int = 200) -> Dict[str, float]:
    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Stream to avoid disk writes; fall back if parquet export is unavailable.
    try:
        ds_iter = load_dataset(
            "osanseviero/code_search_net_parquet",
            "python",
            split=split,
            trust_remote_code=False,
            streaming=True,
        )
    except Exception:
        ds_iter = load_dataset(
            "code_search_net",
            "python",
            split=split,
            trust_remote_code=True,
            streaming=True,
        )

    sampled = list(islice(ds_iter, limit))
    if not sampled:
        raise RuntimeError("No examples loaded for code eval")

    def _extract(example: Dict[str, Any]):
        query = example.get("docstring") or example.get("func_documentation_string") or ""
        code = example.get("code") or example.get("func_code_string") or ""
        return query, code

    pairs = [_extract(ex) for ex in sampled]
    # Filter out empty entries that can appear in streaming slices.
    pairs = [(q, c) for q, c in pairs if q and c]
    if not pairs:
        raise RuntimeError("No non-empty examples for code eval")

    queries, codes = zip(*pairs)

    code_batch = tokenizer(codes, padding=True, truncation=True, max_length=256, return_tensors="pt")
    query_batch = tokenizer(queries, padding=True, truncation=True, max_length=128, return_tensors="pt")

    with torch.no_grad():
        code_emb = model(code_batch["input_ids"].to(device), mask=code_batch["attention_mask"].to(device))["dense"]
        query_emb = model(query_batch["input_ids"].to(device), mask=query_batch["attention_mask"].to(device))["dense"]

    sims = _cosine(query_emb.unsqueeze(1), code_emb.unsqueeze(0))  # (Q, C)

    k5 = min(5, sims.size(1))
    k10 = min(10, sims.size(1))
    top5 = torch.topk(sims, k=k5, dim=1).indices
    top10 = torch.topk(sims, k=k10, dim=1).indices

    hits5 = 0
    hits10 = 0
    mrr10 = 0.0
    for i in range(sims.size(0)):
        if i in top5[i]:
            hits5 += 1
        if i in top10[i]:
            hits10 += 1
            rank = (top10[i] == i).nonzero(as_tuple=False)[0].item() + 1
            mrr10 += 1.0 / rank
    recall5 = hits5 / sims.size(0)
    recall10 = hits10 / sims.size(0)
    mrr10 = mrr10 / sims.size(0)
    return {"code_recall@5": recall5, "code_recall@10": recall10, "code_mrr@10": mrr10}


__all__ = ["run_code_eval"]
