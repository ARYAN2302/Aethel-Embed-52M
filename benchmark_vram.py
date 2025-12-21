import modal
import torch
import torch.cuda
from transformers import AutoTokenizer, AutoModel

from aethel.model.aethel_model import AethelModel

DOC_LEN = 4096
DEVICE = "cuda"


def measure_peak_memory_bge(model, tokenizer) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    text = "test " * DOC_LEN
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=DOC_LEN).to(DEVICE)
    with torch.no_grad():
        _ = model(**inputs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def measure_peak_memory_aethel(model, tokenizer, chunk_tokens: int = 256) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    text = "test " * DOC_LEN
    ids = tokenizer.encode(text, add_special_tokens=False)
    memory_state = None
    for i in range(0, len(ids), chunk_tokens):
        chunk = ids[i : i + chunk_tokens]
        if not chunk:
            continue
        input_ids = torch.tensor(chunk, device=DEVICE).unsqueeze(0)
        mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = model.forward_with_memory(input_ids, mask=mask, memory_state=memory_state)
        memory_state = out.get("memory", None)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers")
    .add_local_dir("aethel", "/root/aethel")
)

app = modal.App("aethel-vram-benchmark")


@app.function(image=image, gpu="A10G", timeout=300)
def run():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    print(f">>> VRAM Efficiency Sweep (Sequence Length: {DOC_LEN}) <<<")

    # Aethel
    a_model = AethelModel(vocab_size=32000, dim=768).to(DEVICE).eval()
    a_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    a_mem = measure_peak_memory_aethel(a_model, a_tokenizer)
    print(f"Aethel-52M Peak VRAM: {a_mem:.2f} MB")

    # BGE-M3
    try:
        b_model = AutoModel.from_pretrained("BAAI/bge-m3").to(DEVICE).eval()
        b_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        b_mem = measure_peak_memory_bge(b_model, b_tokenizer)
        print(f"BGE-M3     Peak VRAM: {b_mem:.2f} MB")
        ratio = b_mem / max(a_mem, 1e-6)
        print(f"\nVerdict: Aethel is {ratio:.1f}x lighter than BGE-M3.")
    except Exception as e:
        print(f"BGE-M3     Error/OOM: {e}")


if __name__ == "__main__":
    app.run()
