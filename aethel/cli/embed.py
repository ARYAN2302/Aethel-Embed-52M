import argparse
from pathlib import Path
import json

import torch
from transformers import AutoTokenizer

from aethel.model.aethel_model import AethelModel


def load_model(checkpoint: str | None, device: str) -> AethelModel:
    model = AethelModel(vocab_size=32000, dim=768).to(device).eval()
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    return model


def embed_text(model: AethelModel, tokenizer, text: str, device: str) -> torch.Tensor:
    batch = tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = model.forward_with_memory(batch["input_ids"], mask=batch.get("attention_mask"), memory_state=None)
    return torch.nn.functional.normalize(out["dense"], p=2, dim=-1).squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="Embed text with Aethel.")
    parser.add_argument("--text", type=str, required=True, help="Text to embed")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to Aethel checkpoint")
    parser.add_argument("--tokenizer", type=str, default="BAAI/bge-base-en-v1.5", help="Tokenizer name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_model(args.checkpoint, device)

    emb = embed_text(model, tokenizer, args.text, device)
    print(json.dumps(emb.cpu().tolist()))


if __name__ == "__main__":
    main()
