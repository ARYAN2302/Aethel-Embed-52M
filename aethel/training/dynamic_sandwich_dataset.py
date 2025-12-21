import random
from typing import Dict, Iterator

import torch
from torch.utils.data import IterableDataset


class DynamicSandwichDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len: int = 8192, filler_repeat: int = 50):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        filler_text = "The quick brown fox jumps over the lazy dog. " * filler_repeat
        self.filler_tokens = self.tokenizer.encode(filler_text, add_special_tokens=False)
        if len(self.filler_tokens) == 0:
            raise ValueError("Filler produced zero tokens; adjust filler_text.")

    def _generate(self) -> Dict[str, torch.Tensor]:
        code_val = random.randint(1000, 9999)
        needle_text = f"The secret verification code is {code_val}."
        needle_tokens = self.tokenizer.encode(needle_text, add_special_tokens=False)

        depth = random.choice([0.0, 0.1, 0.5, 0.9, 1.0, random.random()])
        budget = max(0, self.seq_len - len(needle_tokens) - 2)
        prefix_len = int(budget * depth)
        suffix_len = budget - prefix_len

        input_ids = []
        while len(input_ids) < prefix_len:
            input_ids.extend(self.filler_tokens)
        input_ids = input_ids[:prefix_len]

        input_ids.extend(needle_tokens)

        while len(input_ids) < prefix_len + len(needle_tokens) + suffix_len:
            input_ids.extend(self.filler_tokens)
        input_ids = input_ids[: prefix_len + len(needle_tokens) + suffix_len]

        while len(input_ids) < self.seq_len:
            input_ids.extend(self.filler_tokens)
        input_ids = input_ids[: self.seq_len]

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "query_text": "What is the secret verification code?",
            "needle_text": needle_text,
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        while True:
            yield self._generate()


__all__ = ["DynamicSandwichDataset"]
