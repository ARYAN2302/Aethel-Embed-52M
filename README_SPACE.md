---
title: Aethel-Embed 53M
emoji: 🔮
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🔮 Aethel-Embed: Memory-Augmented Hybrid Embeddings

Aethel is a high-efficiency, long-context embedding model (~53M parameters) featuring:
- **Gated DeltaNet Backbone**: Linear-time complexity for long sequences.
- **TITANS-lite Memory**: Surprise-gated memory for stateful document understanding.
- **Hybrid Heads**: Dense (Matryoshka-sliceable) and Sparse (SPLADE-lite) outputs.

## 🚀 Quick Start

1. **Local Run**:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

2. **Deployment**:
   This Space is configured to run the interactive dashboard. It automatically downloads the model weights from the Hugging Face Hub.

## 📊 Performance
- **VRAM**: 6.6x lower than BGE-M3 at 4k tokens.
- **Context**: Supports up to 8k+ tokens with stateful memory.
- **Size**: Only 53M parameters, making it ideal for edge deployment.
