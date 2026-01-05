---
title: Aethel-Embed Dashboard
emoji: 🔮
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.31.0
app_file: examples/app.py
pinned: false
---

# 🔮 Aethel-Embed Dashboard

This is an interactive demonstration of the **Aethel-Embed** model, a 53M parameter memory-augmented hybrid embedding model.

## Features
- **🧪 Live Playground**: Test sentence similarity and long-context retrieval.
- **📈 Visualization**: Explore the embedding space in 2D.
- **🔍 Similarity Search**: Search through a sample corpus.
- **📊 Benchmarks**: Compare VRAM usage and recall against larger models like BGE-M3.

## Model Architecture
Aethel uses a unique combination of:
- **Gated DeltaNet**: A linear-time backbone for efficient processing.
- **TITANS-lite Memory**: Surprise-gated memory for stateful document understanding.
- **Hybrid Head**: Dense (Matryoshka) and Sparse (SPLADE-lite) outputs.

## Local Setup
To run this dashboard locally:
```bash
git clone https://github.com/ARYAN2302/Aethel-Embed-52M
cd Aethel-Embed-52M
pip install -r requirements.txt
streamlit run examples/app.py
```
