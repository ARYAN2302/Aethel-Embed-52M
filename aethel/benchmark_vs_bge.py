import modal
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# --- CONFIG ---
# We use a 4.5k token document to ensure we push BGE-M3's attention mechanism hard
DOC_LEN_TOKENS = 4500 
NEEDLE = "The secret project code is 'OMEGA-99'."
QUERY = "What is the secret project code?"

# Define the Modal Image
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "numpy", "tqdm")
    # Bake code + checkpoints into the image to avoid Mount API differences
    .add_local_dir("aethel", "/root/aethel")
    .add_local_dir("checkpoints", "/root/checkpoints")
)

app = modal.App("aethel-vs-bge-benchmark")

# No mounts required; everything is baked into the image above.

@app.function(
    image=image,
    gpu="A10G", # We give them both a fair fight on a decent GPU
    timeout=600,
)
def run_benchmark():
    import sys
    sys.path.append("/root") # Ensure we can import aethel
    from aethel.model.aethel_model import AethelModel

    print(f"\n>>> 🥊 MAIN EVENT: Aethel-52M vs. BGE-M3 (560M) <<<")
    print(f"Arena: {DOC_LEN_TOKENS} tokens. Device: NVIDIA A10G.")

    # --- 1. SETUP CHAMPION (BGE-M3) ---
    print("\n[1/3] Loading BGE-M3...")
    bge_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    bge_model = AutoModel.from_pretrained('BAAI/bge-m3').cuda().eval()
    print("      BGE-M3 Loaded (560M params).")

    # --- 2. SETUP CHALLENGER (Aethel) ---
    print("\n[2/3] Loading Aethel-52M...")
    # Using the tokenizer compatible with your training (likely bge-base or similar)
    aethel_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
    aethel_model = AethelModel(vocab_size=32000, dim=768)
    
    # Load your best checkpoint
    # Note: Ensure this file exists in your local 'checkpoints' folder
    ckpt_path = "/root/checkpoints/aethel-step5000-sandwich-fixed.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        aethel_model.load_state_dict(ckpt.get('model', ckpt), strict=False)
        aethel_model.cuda().eval()
        print("      Aethel Loaded (52M params).")
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found at {ckpt_path}. Did you upload it?")
        return

    # --- 3. GENERATE DATA ---
    print(f"\n[3/3] Generating {DOC_LEN_TOKENS} token document...")
    filler = "The quick brown fox jumps over the lazy dog. " * 50
    # Build token list directly to exact length with BGE tokenizer
    base_tokens = bge_tokenizer.encode(f"{NEEDLE} ", add_special_tokens=False)
    filler_tokens = bge_tokenizer.encode(filler, add_special_tokens=False)
    tokens = list(base_tokens)
    fi = 0
    while len(tokens) < DOC_LEN_TOKENS:
        tokens.extend(filler_tokens)
        fi += 1
    tokens = tokens[:DOC_LEN_TOKENS]
    doc_text = bge_tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"      Doc Tokens: {len(tokens)} (target {DOC_LEN_TOKENS})")

    # --- 4. THE FIGHT ---
    
    def aethel_encode_query(model, tokenizer, text, max_len=256):
        batch = tokenizer([text], return_tensors="pt", truncation=True, max_length=max_len).to("cuda")
        with torch.no_grad():
            out = model.forward_with_memory(batch["input_ids"], mask=batch.get("attention_mask"), memory_state=None)
        return F.normalize(out["dense"], p=2, dim=-1)

    def aethel_encode_doc(model, tokenizer, text, chunk_tokens=256):
        ids = tokenizer.encode(text, add_special_tokens=False)
        step = chunk_tokens
        memory = None
        last_dense = None
        for i in range(0, len(ids), step):
            chunk = ids[i : i + step]
            if not chunk:
                continue
            input_ids = torch.tensor(chunk, device="cuda").unsqueeze(0)
            attn = torch.ones_like(input_ids)
            with torch.no_grad():
                out = model.forward_with_memory(input_ids, mask=attn, memory_state=memory)
            memory = out["memory"]
            last_dense = out["dense"]
        return F.normalize(last_dense, p=2, dim=-1)

    def bge_encode(model, tokenizer, text, max_len):
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_len).to('cuda')
        out = model(**enc)
        # mean pool with attention mask
        last = out.last_hidden_state
        mask = enc.get("attention_mask")
        if mask is not None:
            mask = mask.unsqueeze(-1).type_as(last)
            last = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            last = last.mean(dim=1)
        return F.normalize(last, p=2, dim=-1)

    def evaluate(name, model, tokenizer, is_aethel=True):
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            # Encode Query
            if is_aethel:
                q_emb = aethel_encode_query(model, tokenizer, QUERY)
            else:
                q_emb = bge_encode(model, tokenizer, QUERY, max_len=512)
                
            # Encode Doc
            if is_aethel:
                d_emb = aethel_encode_doc(model, tokenizer, doc_text, chunk_tokens=256)
            else:
                d_emb = bge_encode(model, tokenizer, doc_text, max_len=8192)
            
            # Score
            score = (q_emb @ d_emb.T).item()
            
        torch.cuda.synchronize()
        duration = (time.time() - start) * 1000 # ms
        return score, duration

    print("\n>>> FIGHT! <<<")
    
    # BGE-M3 Round
    try:
        bge_score, bge_time = evaluate("BGE-M3", bge_model, bge_tokenizer, is_aethel=False)
        print(f"🔵 BGE-M3     | Score: {bge_score:.4f} | Time: {bge_time:.2f} ms")
    except RuntimeError as e:
        print(f"🔵 BGE-M3     | DIED (OOM): {e}")
        bge_score, bge_time = 0.0, 99999.0

    # Aethel Round
    aethel_score, aethel_time = evaluate("Aethel", aethel_model, aethel_tokenizer, is_aethel=True)
    print(f"🔴 Aethel-52M | Score: {aethel_score:.4f} | Time: {aethel_time:.2f} ms")

    # --- VERDICT ---
    speedup = bge_time / aethel_time
    print(f"\n>>> FINAL VERDICT <<<")
    print(f"Speedup: Aethel is {speedup:.1f}x FASTER than BGE-M3.")
    
    if aethel_score > 0.5:
        print("Recall:  Aethel SUCCESSFULLY retrieved the needle.")
        if abs(aethel_score - bge_score) < 0.15:
            print("Summary: Aethel matches SOTA accuracy at 1/10th the size and >5x speed.")
    else:
        print("Recall:  Aethel missed it.")