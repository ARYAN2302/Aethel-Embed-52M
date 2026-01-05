# Aethel Deployment Guide

This guide outlines how to deploy the Aethel embedding model and the interactive dashboard.

## 1. Model API Deployment (FastAPI)

To serve embeddings via an API, you can use FastAPI.

### Prerequisites
```bash
pip install fastapi uvicorn torch transformers
```

### API Implementation (`api.py`)
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from aethel.model.aethel_model import AethelModel
import torch.nn.functional as F

app = FastAPI(title="Aethel Embedding API")

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AethelModel(vocab_size=32000, dim=768).to(device).eval()
checkpoint = torch.load("checkpoints/aethel-step5000.pt", map_location=device)
model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed(request: EmbedRequest):
    try:
        batch = tokenizer([request.text], return_tensors="pt", truncation=True, max_length=512)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model.forward_with_memory(batch["input_ids"], mask=batch.get("attention_mask"))
        
        embedding = F.normalize(out["dense"], p=2, dim=-1).cpu().tolist()[0]
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 2. Dashboard Deployment (Streamlit)

The dashboard can be deployed to Streamlit Cloud, Hugging Face Spaces, or any VPS.

### Local/VPS Deployment
```bash
streamlit run examples/app.py
```

### Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "examples/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 3. Cloud Deployment (Modal)

For serverless deployment with GPU support, Modal is recommended.

```python
import modal

stub = modal.Stub("aethel-api")
image = modal.Image.debian_slim().pip_install("torch", "transformers", "fastapi", "uvicorn")

@stub.function(image=image, gpu="A10G", mounts=[modal.Mount.from_local_dir("./aethel", remote_path="/root/aethel")])
@modal.asgi_app()
def fastapi_app():
    from api import app
    return app
```

## 4. Optimization for Production

- **Quantization:** Use `aethel/export/quantize.py` to convert the model to INT8 or ONNX for faster CPU inference.
- **Batching:** Update the API to handle lists of strings for higher throughput.
- **Caching:** Use Redis or similar to cache embeddings for frequent queries.
