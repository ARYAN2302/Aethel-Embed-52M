"""Generate a needle-in-a-haystack heatmap for Effective Context Length."""

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from transformers import AutoTokenizer

from aethel.model.aethel_model import AethelModel
from aethel.eval.long_context_eval import _cosine, _encode_long_context, _encode_query


# Default sweep parameters
DEFAULT_CONTEXT_LENGTHS = [1024, 2048, 4096, 8192, 12000]
DEFAULT_DEPTHS = [0.0, 0.25, 0.50, 0.75, 1.0]
DEFAULT_CODES = ["Blueberry", "Sky", "Ocean", "Mountain", "River"]
NEEDLE_TEMPLATE = "The secret code is {code}."
QUERY_TEMPLATE = "What is the secret code?"
HAYSTACK_TEXT = "The quick brown fox jumps over the lazy dog. " * 50


def _parse_list(raw: str, cast=int) -> List[int]:
    return [cast(x) for x in raw.split(",") if x]


def build_context_tokens(tokenizer, context_len: int, depth_pct: float, needle_tokens: List[int]) -> List[int]:
    filler_tokens = tokenizer.encode(HAYSTACK_TEXT, add_special_tokens=False)
    if len(filler_tokens) == 0:
        raise ValueError("Filler text produced zero tokens; update HAYSTACK_TEXT")

    total_budget = context_len
    pre_budget = max(0, min(total_budget - len(needle_tokens), int(depth_pct * (total_budget - len(needle_tokens)))))

    context_tokens: List[int] = []
    # Fill pre-needle tokens
    while len(context_tokens) < pre_budget:
        context_tokens.extend(filler_tokens)
    context_tokens = context_tokens[:pre_budget]

    # Insert needle
    context_tokens.extend(needle_tokens)

    # Fill remainder
    while len(context_tokens) < total_budget:
        context_tokens.extend(filler_tokens)
    return context_tokens[:total_budget]


def _robust_encode_chunks(model, tokenizer, text: str, device: torch.device, safe_window: int = 2048, overlap: int = 256) -> torch.Tensor:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= safe_window:
        chunk_embs, _ = _encode_long_context(model, tokenizer, text, chunk_tokens=safe_window, device=device)
        return chunk_embs

    chunk_embs = []
    step = max(1, safe_window - overlap)
    for start in range(0, len(tokens), step):
        window = tokens[start : start + safe_window]
        if not window:
            continue
        chunk_text = tokenizer.decode(window, skip_special_tokens=True)
        bos = tokenizer.bos_token or ""
        forced_start_text = f"{bos} {chunk_text}".strip()

        if hasattr(model, "reset_memory"):
            model.reset_memory()

        embeds, _ = _encode_long_context(model, tokenizer, forced_start_text, chunk_tokens=safe_window, device=device)
        if embeds.numel() == 0:
            continue
        chunk_embs.append(embeds)
    if not chunk_embs:
        return torch.empty(0, device=device)
    return torch.cat(chunk_embs, dim=0)


def run_needle_test(model, tokenizer, context_len: int, depth_pct: float, chunk_tokens: int, device: torch.device, codes: Iterable[str], use_robust: bool = False) -> int:
    needle_code = np.random.choice(list(codes))
    needle = NEEDLE_TEMPLATE.format(code=needle_code)
    needle_tokens = tokenizer.encode(needle, add_special_tokens=False)

    context_tokens = build_context_tokens(tokenizer, context_len, depth_pct, needle_tokens)
    full_text = tokenizer.decode(context_tokens)

    # Encode document into chunk embeddings; MaxSim scoring
    if use_robust:
        chunk_embs = _robust_encode_chunks(model, tokenizer, full_text, device=device, safe_window=max(chunk_tokens, 2048))
    else:
        chunk_embs, _ = _encode_long_context(model, tokenizer, full_text, chunk_tokens=chunk_tokens, device=device)
    if chunk_embs.numel() == 0:
        return 0

    query_emb = _encode_query(model, tokenizer, QUERY_TEMPLATE, max_len=chunk_tokens, device=device)
    sims = _cosine(query_emb, chunk_embs).view(-1)

    insertion_idx = max(0, min(context_len - len(needle_tokens), int(depth_pct * (context_len - len(needle_tokens)))))
    target_chunk_idx = insertion_idx // chunk_tokens

    top_idx = torch.argmax(sims).item()
    is_hit = abs(top_idx - target_chunk_idx) <= 1
    return 1 if is_hit else 0


def load_model(model_path: Path, device: torch.device) -> AethelModel:
    ckpt = torch.load(model_path, map_location=device)
    model = AethelModel(vocab_size=32000, dim=768).to(device)
    missing, unexpected = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    if missing:
        print(f"Loaded with missing keys (expected for older checkpoints): {missing}")
    if unexpected:
        print(f"Loaded with unexpected keys: {unexpected}")
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Needle-in-a-haystack heatmap")
    parser.add_argument("--model-path", type=Path, default=Path("checkpoints/aethel-step5000.pt"))
    parser.add_argument("--tokenizer", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--context-lengths", type=str, default=",".join(str(x) for x in DEFAULT_CONTEXT_LENGTHS))
    parser.add_argument("--depths", type=str, default=",".join(str(d) for d in DEFAULT_DEPTHS))
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("aethel_heatmap.png"))
    parser.add_argument("--robust", action="store_true", help="Use tiled robust encoding (safe window + max over chunks)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    context_lengths = _parse_list(args.context_lengths, cast=int)
    depths = _parse_list(args.depths, cast=float)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_model(args.model_path, device=device)

    results = np.zeros((len(depths), len(context_lengths)), dtype=float)

    print("Starting Needle-in-a-Haystack scan...")
    for j, context_len in enumerate(context_lengths):
        for i, depth in enumerate(depths):
            desc = f"len={context_len} depth={int(depth*100)}%"
            print(f"Testing {desc}")
            score = run_needle_test(model, tokenizer, context_len, depth, args.chunk_tokens, device, DEFAULT_CODES, use_robust=args.robust)
            results[i, j] = score

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        results,
        annot=True,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        xticklabels=context_lengths,
        yticklabels=[f"{int(d*100)}%" for d in depths],
    )
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Needle Depth")
    plt.title("Aethel: Long-Context Recall Heatmap")
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Heatmap saved to {args.output}")


if __name__ == "__main__":
    main()
