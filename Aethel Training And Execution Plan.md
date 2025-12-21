# Aethel — Training & Execution Plan

Goal: ship a ~100M-parameter, long-context, memory-augmented hybrid embedding model with dense + sparse outputs, trained via multi-teacher distillation under a $30–$50 budget on a single L40S/A100-class GPU.

---

## 0) Deliverables
- Trainable codebase with DeltaNet stack, sliding-window attention, TITANS-lite memory, and hybrid dense+sparse head.
- Data pipeline for ~20M tokens (text, long docs, code, instruction retrieval).
- Distillation loop with BGE-M3 + Qwen embeddings + code teacher.
- Evaluation harness (mini-MTEB, long-context recall, code retrieval).
- Export + quantization path (ONNX + INT8/INT4) and backward-compatible adapters.

---

## 1) Model Configuration (v1)
- d_model: 512; dense head: 768 (Matryoshka slices 768→512→256→128).
- DeltaNet layers: 6 (fast-weight + local conv + gated residual).
- Sliding-window attention layers: 2 (window=256, 8 heads).
- Titans-lite memory: 1024d with surprise-gated update, momentum, decay.
- Sparse head: SPLADE-lite over vocab 5k (top-k ≈ 200 recommended).
- Training context: 1k (optionally 2k); inference extension: 8k–32k via RoPE/ALiBi scaling.
- Parameter target: ~100M.

---

## 2) Dataset Plan (~20M tokens)

| Data Type                         | Tokens | Purpose                        |
| --------------------------------- | ------ | ------------------------------ |
| General text (Wiki, Books)        | ~6M    | Baseline semantics             |
| Long-document synthetic data      | ~5M    | Memory + long-context training |
| Code + docstring                  | ~3M    | Code retrieval                 |
| Instruction-style retrieval pairs | ~3M    | RAG alignment                  |
| Sparse-head supervision           | —      | Derived from teachers          |

Acquisition checklist:
- Pull cleaned Wikipedia/Books slices; ensure language/tokenization consistency.
- Generate long-doc synthetic sequences (e.g., concatenated chapters with section markers) for memory/recall stress.
- Curate code-docstring pairs from permissive sources; deduplicate; keep line-level spans.
- Build retrieval pairs (q, doc, negatives) for InfoNCE; include code and text mixes.
- Ensure sparse targets derived from teacher sparse logits.

---

## 3) Teacher Models (distillation)
- Dense: BGE-M3 preferred; fallback BGE-base; Qwen2.5/Qwen3 embeddings for long-context semantics.
- Sparse: BGE-M3 sparse head; optional SPLADE.
- Code: lightweight Codestral/Code-LLaMA-small or CodeBERT embeddings.

Fetch checklist:
- Cache teacher checkpoints locally.
- Wrap inference in half precision where safe; ensure deterministic outputs for logging.

---

## 4) Losses and Weights
- Dense distillation: cosine / MSE on dense embeddings.
- Contrastive InfoNCE: positives from paired data; hard negatives from in-batch + mined.
- Matryoshka slicing: enforce consistency on {128,256,512,768} slices.
- Sparse head: BCE or KL against teacher sparse logits (top-k masked).
- Memory consistency: align Titans-lite memory state to pooled teacher embedding.

Overall loss: $L = 1.0 L_{dense} + 0.5 L_{infoNCE} + 0.5 L_{matryoshka} + 0.3 L_{sparse} + 0.2 L_{memory}$.

---

## 5) Hyperparameters (baseline)
- batch_size: 64; seq_len: 1024 ⇒ tokens/step ≈ 65k.
- steps/epoch: ~305 for 20M tokens; epochs: 8 ⇒ ~2440 steps.
- Optimizer: AdamW (lr 2e-4, betas 0.9/0.98, wd 0.01).
- LR schedule: warmup 400 steps, cosine decay to 2e-5.
- Regularization: dropout minimal (SSM-friendly), gradient clip 1.0, label smoothing 0 for sparse BCE.
- Mixed precision: bf16/amp enabled.

---

## 6) Training Loop Plan
- Build dataloaders with dynamic mixing ratios per bucket (text/long/code/inst) and curriculum for early stability.
- Teacher forward pass (no grad), cache logits/embeddings; optionally precompute for static sets.
- Student forward: embeddings → DeltaNet stack → sliding attention → pooling + memory update → hybrid head.
- Compute losses per section 4; log per-term scalars.
- Optimizer step with gradient accumulation if VRAM-bound; EMA optional.
- Checkpoint every N steps (e.g., 200) with optimizer, scheduler, and memory state.
- Logging: step time, throughput, memory saturation, per-loss, dense/sparse norms, slice quality.

---

## 7) Infrastructure, Cost, and Schedule
- Target hardware: L40S ($1.95/h), A100 40GB ($2.10/h), A100 80GB ($2.50/h).
- Expected wall-clock: 12–20 hours single-GPU; budget $30–$50 per spec.
- Storage: teacher checkpoints + datasets (~tens of GB). Ensure fast local scratch.

Execution to Friday (T=delivery):
- T-4/T-3: finalize data pulls, tokenize, shard; validate dataloader throughput; smoke-test teachers.
- T-3/T-2: wire training loop, loss plumbing, logging; run 1–2 tiny steps for shape sanity.
- T-2/T-1: run main 12–20h training; monitor loss curves; adjust weights if sparse/dense imbalance.
- T-1/T: run evals (mini-MTEB subset, long-context recall, code retrieval); select checkpoint; export and quantize.

---

## 8) Evaluation Plan
- Mini-MTEB subset: STS-lite, retrieval-lite, clustering-lite; report Spearman/MRR/NMI.
- Long-context recall: synthetic long-doc QA and section recall; measure hit rate at k.
- Code retrieval: CoIR-lite; measure MRR/NDCG; include identifier-heavy queries.
- Ablations: memory on/off, sparse head on/off, attention vs DeltaNet-only.

---

## 9) Export & Packaging
- ONNX export of encoder with fixed positional scaling knobs.
- Quantization: PTQ/AWQ-style INT8/INT4; calibrate on mixed code/text batches.
- Adapters: backward-compatible dense adapters for vector DB drop-in; provide slice mapping for 768→128.
- CLI stubs: `python -m aethel.export.export_onnx`, `python -m aethel.export.quantize` (to be implemented).

---

## 10) Risks & Mitigations
- Sparse head underperforms: raise sparse weight, increase teacher top-k, or add SPLADE teacher.
- Memory collapse/instability: add small L2 on memory norm; gate updates with surprise threshold.
- Long-context extrapolation drift: use RoPE scaling; add few-shot 2k training; test at 8k early.
- Code retrieval lagging: increase code-docstring fraction; add identifier masking augmentation.
- Budget creep: reduce batch, freeze teachers to cached embeddings where possible, prefer L40S.

---

## 11) Execution Checklist
- [ ] Datasets downloaded, tokenized, and mixed to ~20M tokens with per-bucket ratios.
- [ ] Teachers cached; inference wrappers validated; sparse logits available.
- [ ] Student forward shapes validated end-to-end; loss terms log correctly.
- [ ] Training run launched with logging + checkpoints; monitor loss balance and throughput.
- [ ] Evaluation suite executed; checkpoint selected.
- [ ] Exported ONNX + quantized artifacts + adapters produced.
- [ ] README updated with usage examples and results.# Aethel — Full Training Plan, Execution Checklist, and Roadmap

This document consolidates all planning from our discussion into a single, clean, structured markdown file. It includes:

* Full training plan
* Hyperparameters
* Architecture configuration
* Execution checklist
* Dataset plan
* Loss functions
* GPU cost strategy
* Roadmap to Friday launch

---

# 1. Aethel Model Configuration (v1)

### **Core Dimensions**

* **d_model:** 512
* **DeltaNet layers:** 6
* **Local sliding attention layers:** 2
* **heads:** 8
* **feedforward dimension:** ~1536
* **memory_dim (Titans-lite):** 1024
* **dense embedding dimension:** 768
* **sparse vocabulary size:** 5000 (top-k ~200 recommended)

### **Context Length Strategy**

* Training at **1024 tokens** for cost efficiency.
* Optional short training at **2048 tokens**.
* Inference-time extension to **8k–32k** via RoPE/ALiBi scaling.

---

# 2. Dataset Plan

### **Total Required Tokens:** **~20M tokens**

### **Recommended Composition**

| Data Type                         | Tokens | Purpose                        |
| --------------------------------- | ------ | ------------------------------ |
| General text (Wiki, Books)        | ~6M    | Baseline semantics             |
| Long-document synthetic data      | ~5M    | Memory + long-context training |
| Code + docstring                  | ~3M    | Code retrieval                 |
| Instruction-style retrieval pairs | ~3M    | RAG alignment                  |
| Sparse-head supervision           | —      | Derived from teachers          |

### **Why so small?**

Distillation + efficient SSM/DeltaNet compression means we avoid expensive LM-style training.

---

# 3. Training Hyperparameters

### **Batching & Epochs**

* **batch_size:** 64
* **seq_len:** 1024
* **tokens/step:** 64 × 1024 ≈ 65,536
* **steps per epoch:** ~305
* **total epochs:** 8
* **total steps:** ~2440

### **Optimizer**

* **AdamW**

  * lr = 2e-4
  * betas = (0.9, 0.98)
  * weight_decay = 0.01

### **LR Schedule**

* warmup_steps = 400
* cosine decay to 2e-5

---

# 4. Teacher Models

### **Dense Teacher Options**

* `BAAI/bge-base-en-v1.5`
* `BAAI/bge-m3` (preferred if using sparse too)

### **Sparse Teacher**

* BGE-M3 sparse head
* SPLADE (optional)

### **Code Teacher**

* Lightweight CodeBERT or Mini-Codestral for embeddings

---

# 5. Loss Functions (Per Batch)

### **1. Dense Distillation Loss**

Match student dense embedding to teacher embedding.

### **2. Contrastive InfoNCE**

Uses positive/negative pairs.

### **3. Matryoshka Slicing Loss**

Student slices: [128d, 256d, 512d, 768d]

### **4. Sparse Head Loss**

BCE or KL on teacher sparse logits.

### **5. Memory Consistency Loss**

Memory state ≈ pooled teacher embedding.

### **Suggested Weights**

```
L = 1.0 * L_dense_distill
  + 0.5 * L_contrastive
  + 0.5 * L_matryoshka
  + 0.3 * L_sparse
  + 0.2 * L_memory
```

---

# 6. GPU Cost & Time Estimation

### **Token FLOPs**

DeltaNet embedding model ≈ 500M FLOPs/token.

### **20M token pass:**

= ~1e16 FLOPs = ~10 TFLOPs compute

Modern GPUs (A100, L40S, H100) finish this extremely quickly.

### **Estimated Training Time**

| GPU                  | Total Time | Cost  |
| -------------------- | ---------- | ----- |
| L40S ($1.95/hr)      | 3–6h       | $8–12 |
| A100 40GB ($2.10/hr) | 4–6h       | $9–14 |
| A100 80GB ($2.50/hr) | 3–5h       | $7–12 |
| H100 ($3.95/hr)      | 2–4h       | $8–15 |

### **Conclusion:** Aethel v1 fits **comfortably** in a **$10–$15** compute budget.

And you get **$30 free per month** from Modal.

---



* Perform mini-runs of:

  * Semantic similarity (STS-lite)
  * Retrieval examples
  * Code search mini-tests
* Finalize README, diagrams, blog outline.
* Polish repo for release.

---


---

# 9. Expected Outcomes

### **Aethel v1 achieves:**

* Long-context awareness (8k–32k at inference)
* Hybrid sparse + dense embeddings
* Titans-lite memory enhancement
* Matryoshka slicing for scalable vector DB costs
* Strong performance despite only 100M params
* Extremely low inference cost

This absolutely qualifies as a **novel architecture** and a **real research contribution**.

---

# 10. Next Steps

When ready, proceed to:

* Implement final model code
* Build training loop
* Prepare evaluation utilities

If further help is needed with any code file (DeltaNet, memory, hybrid head, dataset builder, or training script), request the component and it will be generated.
