import os
import torch
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer
import shutil

def upload_model(repo_id, checkpoint_path, local_model_dir="aethel"):
    api = HfApi()
    
    print(f"Creating/Checking repository: {repo_id}")
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    # 1. Upload the model code
    print(f"Uploading model code from {local_model_dir}...")
    api.upload_folder(
        folder_path=local_model_dir,
        repo_id=repo_id,
        path_in_repo="aethel",
    )

    # 2. Upload the checkpoint
    print(f"Uploading checkpoint {checkpoint_path}...")
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo="aethel-step5000.pt",
        repo_id=repo_id,
    )

    # 3. Upload the tokenizer (we use BGE-base-en-v1.5 as base)
    print("Uploading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    tokenizer.save_pretrained("./temp_tokenizer")
    api.upload_folder(
        folder_path="./temp_tokenizer",
        repo_id=repo_id,
        path_in_repo="tokenizer",
    )
    shutil.rmtree("./temp_tokenizer")

    # 4. Upload a basic README
    readme_content = f"""---
language: en
tags:
- sentence-similarity
- feature-extraction
- aethel
- deltanet
- titans
---

# Aethel-Embed (53M)

Aethel is a memory-augmented hybrid embedding model designed for efficiency and long-context performance.

## Model Details
- **Parameters**: ~53M
- **Architecture**: Gated DeltaNet (6 layers) + Sliding Window Attention + TITANS-lite Memory
- **Embedding Dimension**: 768 (Matryoshka-capable)
- **Context Length**: Optimized for long-context retrieval

## Usage
This model requires the `aethel` library code included in this repository.

```python
import torch
from aethel.model.aethel_model import AethelModel
from transformers import AutoTokenizer

# Load model and tokenizer
model = AethelModel(vocab_size=32000, dim=768)
checkpoint = torch.load("aethel-step5000.pt", map_location="cpu")
model.load_state_dict(checkpoint.get("model", checkpoint))

tokenizer = AutoTokenizer.from_pretrained("tokenizer/")
```
"""
    with open("README_HF.md", "w") as f:
        f.write(readme_content)
    
    api.upload_file(
        path_or_fileobj="README_HF.md",
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    os.remove("README_HF.md")

    print(f"Successfully uploaded to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    # Replace with your desired repo name
    REPO_ID = "aryan2302/Aethel-Embed-53M" 
    CHECKPOINT = "checkpoints/aethel-step5000.pt"
    
    if os.path.exists(CHECKPOINT):
        upload_model(REPO_ID, CHECKPOINT)
    else:
        print(f"Error: Checkpoint not found at {CHECKPOINT}")
