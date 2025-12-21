from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, interleave_datasets
from transformers import AutoTokenizer


@dataclass
class SplitSpec:
    dataset_name: str
    split: str
    text_field: str
    weight: float = 1.0
    streaming: bool = False
    limit: Optional[int] = None
    config_name: Optional[str] = None


def _load_and_tokenize(spec: SplitSpec, tokenizer: AutoTokenizer, seq_len: int):
    load_kwargs = {"path": spec.dataset_name, "split": spec.split}
    if spec.config_name:
        load_kwargs["name"] = spec.config_name
    try:
        ds = load_dataset(**load_kwargs)
    except ValueError as exc:
        # Fall back to full split if fractional/percent slicing is unsupported.
        if "Unrecognized instruction format" in str(exc):
            fallback_split = "train" if spec.split.startswith("train") else spec.split.split(":" ,1)[0]
            load_kwargs["split"] = fallback_split
            ds = load_dataset(**load_kwargs)
        else:
            raise

    # Drop empty/whitespace-only examples to avoid all-pad batches.
    ds = ds.filter(lambda ex: bool(ex.get(spec.text_field, "").strip()))

    # Drop non-text columns to avoid feature alignment issues when interleaving.
    keep_cols = {spec.text_field}
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    if spec.limit and not spec.streaming:
        limit = min(spec.limit, len(ds))
        ds = ds.select(range(limit))

    def _tokenize(example):
        text = example.get(spec.text_field, "")
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            return_attention_mask=True,
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    ds = ds.map(_tokenize, remove_columns=[spec.text_field], batched=False)
    return ds


def _collate_fn(batch: Iterable[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def build_datasets(config: Any) -> Tuple[DatasetDict, DataLoader]:
    tokenizer_name = getattr(config, "tokenizer_name", "sentence-transformers/all-MiniLM-L6-v2")
    seq_len = getattr(config, "seq_len", 1024)
    batch_size = getattr(config, "batch_size", 8)
    specs: List[SplitSpec] = getattr(config, "splits", [])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    if not specs:
        specs = [
            SplitSpec(dataset_name="wikipedia", split="train[:1%]", text_field="text", weight=1.0, limit=2000),
        ]

    datasets = [_load_and_tokenize(spec, tokenizer, seq_len) for spec in specs]

    if len(datasets) == 1:
        ds = datasets[0]
    else:
        weights = [s.weight for s in specs]
        total = sum(weights)
        probs = [w / total for w in weights]
        ds = interleave_datasets(datasets, probabilities=probs, stopping_strategy="first_exhausted")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)
    return DatasetDict({"train": ds}), loader


__all__ = ["build_datasets", "SplitSpec"]
