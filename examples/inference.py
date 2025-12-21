import torch
from transformers import AutoTokenizer

from aethel.model.aethel_model import AethelModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

    # Load from local checkpoint; swap with HF path when published.
    ckpt = torch.load("checkpoints/aethel-step5000.pt", map_location=device)
    model = AethelModel(vocab_size=32000, dim=768).to(device).eval()
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    text = "hello world"
    batch = tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        out = model.forward_with_memory(batch["input_ids"], mask=batch.get("attention_mask"), memory_state=None)
    emb = torch.nn.functional.normalize(out["dense"], p=2, dim=-1)
    print(emb.squeeze(0))


if __name__ == "__main__":
    main()
