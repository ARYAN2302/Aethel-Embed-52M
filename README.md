# Aethel

Aethel is a ~53M-parameter, long-context, memory-augmented hybrid embedding model for RAG, code retrieval, and document understanding. It combines a gated DeltaNet backbone, sliding-window attention, a TITANS-lite memory module, and dual dense+sparse heads (Matryoshka 768→128 plus SPLADE-lite top-k keywords) to deliver compact, production-ready embeddings.

---

## 🤗 Hugging Face

- **Model Card**: [aryan2302/Aethel-Embed-53M](https://huggingface.co/aryan2302/Aethel-Embed-53M)
- **Live Demo**: [Aethel-Embed Dashboard](https://huggingface.co/spaces/aryan2302/Aethel-embed-demo)

---

## 🎯 Quick Demo

Launch the interactive dashboard to explore the embedding space:

```bash
# Clone and setup
git clone https://github.com/ARYAN2302/Aethel-Embed-52M.git
cd Aethel-Embed-52M

# Install dependencies
pip install -r requirements.txt

# Launch the demo dashboard
streamlit run examples/app.py
```

Then open **http://localhost:8501** to explore:
- 📊 Interactive 2D embedding space visualization
- 🔍 Similarity search demo
- 📈 Benchmark comparisons
- 🏗️ Model architecture overview

---

## 🎮 Dashboard Demo Features

The Streamlit dashboard includes:

### Embedding Space Visualization
- **2D Scatter Plot** - Explore how concepts cluster in embedding space
- **Color by Category** - AI/ML, Data, Development, Infrastructure
- **Multiple Methods** - PCA, t-SNE, UMAP projection options
- **Interactive** - Zoom, pan, and hover for details

### Similarity Search Demo
- **Query Input** - Type any search query
- **Cosine Similarity** - Real-time similarity computation
- **Ranking Results** - Top matches with similarity scores
- **Visual Progress** - Similarity bar indicators

### Benchmark Comparisons
- **VRAM Usage** - Aethel vs BGE-M3 vs MiniLM
- **Long-Context Recall** - Performance at 4.5k tokens
- **Parameter Efficiency** - 53M vs 560M parameters
- **Key Metrics** - 6.6× lower VRAM, better recall

### Model Architecture
- **Visual Flow Diagram** - Token → DeltaNet → Attention → Memory → Head
- **Component Details** - Descriptions of each module
- **Memory Visualization** - How TITANS-lite works

---

## Model Architecture

```
Input Tokens → Token Embedding → Gated DeltaNet → Sliding Window → TITANS Memory → Hybrid Head
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Token Embedding** | Learned embeddings with 1.5x scaling |
| **Gated DeltaNet** | 6-layer transformer with gating mechanism |
| **Sliding Window Attention** | Local context with configurable window |
| **TITANS-Lite Memory** | External memory for long-context modeling |
| **Hybrid Head** | Dual dense (Matryoshka) + sparse (SPLADE-lite) outputs |

---

## Repository Layout

```
aethel/
  model/          # Core architecture components
  training/       # Datasets, distillation, and training loop
  eval/           # Mini-MTEB, long-context, and code evals
  export/         # Export, quantization, adapters
examples/
  app.py          # Interactive dashboard
  inference.py    # Minimal inference example
```

---

## Quickstart

- Install (editable): `pip install -e .`
- Embed from CLI: `python -m aethel.cli.embed --text "hello world" --checkpoint checkpoints/aethel-step5000.pt`
- Minimal inference: `python examples/inference.py`
- Dashboard: `streamlit run examples/app.py`
- Long-context benchmark (Modal): `modal run benchmark_vs_bge.py`
- VRAM sweep (Modal): `modal run benchmark_vram.py`

---

## Evaluations

Built-in: mini STS (STS-B dev), synthetic long-context recall, code retrieval.

## Comparisons

- vs BGE-M3 (560M): Aethel ~6.6× lower VRAM on 4k tokens; competitive recall at 4.5k tokens.
- vs BGE-base / MiniLM: Aethel provides significantly better long-context recall and lower latency.

## Benchmark Results

| Metric | Aethel (~53M) | BGE-M3 (560M) | BGE-base |
|--------|---------------|---------------|----------|
| VRAM (4k tokens) | 2.1 GB | 14.0 GB | 8.5 GB |
| Long-Context Recall (4.5k) | 92.5% | 88.3% | 85.1% |
| Parameters | ~53M | ~560M | ~110M |

---

## Guardrails / Known Limitations

- English-only; not tuned for factual QA.
- Memory can hallucinate on noisy inputs; sparse head is basic.
- No multi-GPU/distributed training path yet.

---

## License

MIT License

---

## 👤 Author

**Aryan** - [GitHub](https://github.com/ARYAN2302)

---

**Star ⭐ the repo if you find it useful!**
