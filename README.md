# Aethel

Aethel is a ~100M-parameter, long-context, memory-augmented hybrid embedding model for RAG, code retrieval, and document understanding. It combines a gated DeltaNet backbone, sliding-window attention, a TITANS-lite memory module, and dual dense+sparse heads (Matryoshka 768→128 plus SPLADE-lite top-k keywords) to deliver compact, production-ready embeddings.

## Repository Layout

```
aethel/
  model/          # Core architecture components
  training/       # Datasets, distillation, and training loop
  eval/           # Mini-MTEB, long-context, and code evals
  export/         # Export, quantization, adapters
```

## Quickstart

- Install (editable): `pip install -e .`
- Embed from CLI: `python -m aethel.cli.embed --text "hello world" --checkpoint checkpoints/aethel-step5000.pt`
- Minimal inference: `python examples/inference.py`
- Long-context benchmark (Modal): `modal run benchmark_vs_bge.py`
- VRAM sweep (Modal): `modal run benchmark_vram.py`

## Evaluations

Built-in: mini STS (STS-B dev), synthetic long-context recall, code retrieval.

## Comparisons

- vs BGE-M3 (560M): Aethel ~6.6× lower VRAM on 4k tokens; competitive recall at 4.5k tokens.
- vs BGE-base / MiniLM: Aethel provides significantly better long-context recall and lower latency.

## Guardrails / Known Limitations

- English-only; not tuned for factual QA.
- Memory can hallucinate on noisy inputs; sparse head is basic.
- No multi-GPU/distributed training path yet.
