from typing import Any, Dict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double()
    y = y.double()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt((vx ** 2).sum() * (vy ** 2).sum()).clamp(min=1e-9)
    return (vx * vy).sum().item() / denom.item()


def run_mini_mteb(model: Any, tokenizer_name: str = "BAAI/bge-base-en-v1.5", split: str = "dev[:500]") -> Dict[str, float]:
    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Map old 'validation' naming to the dataset's 'dev' split.
    if split.startswith("validation"):
        split = split.replace("validation", "dev", 1)
    ds = load_dataset("stsb_multi_mt", name="en", split=split)

    # Ensure inputs are plain strings (datasets can yield non-str types).
    s1 = [str(x) for x in ds["sentence1"]]
    s2 = [str(x) for x in ds["sentence2"]]
    sent1 = tokenizer(s1, padding=True, truncation=True, max_length=256, return_tensors="pt")
    sent2 = tokenizer(s2, padding=True, truncation=True, max_length=256, return_tensors="pt")

    with torch.no_grad():
        emb1 = model(sent1["input_ids"].to(device), mask=sent1["attention_mask"].to(device))["dense"]
        emb2 = model(sent2["input_ids"].to(device), mask=sent2["attention_mask"].to(device))["dense"]
        sims = _cosine(emb1, emb2).cpu()

    labels = torch.tensor(ds["similarity_score"]) / 5.0  # normalize
    pearson = _pearson(sims, labels)
    return {"stsb_pearson": pearson}


__all__ = ["run_mini_mteb"]
