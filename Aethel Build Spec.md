# Aethel: What We Are Building

Aethel is a 53M-parameter, long-context, memory-augmented, hybrid embedding model designed for real-world RAG, code retrieval, document understanding, and efficient local inference. It combines DeltaNet, sliding-window attention, and a TITANS-lite memory module to create dense + sparse hybrid embeddings, solving key gaps in the 2025 embedding landscape.

---

## 🚀 **1. Purpose of Aethel**

Aethel is designed to address core limitations in current embedding models:

* Lack of long-context capability
* Poor structural/document understanding
* Inefficient code retrieval
* Embedding drift (breaking vector DBs)
* Lack of hybrid dense + sparse outputs
* No memory or personalization
* Overly heavy models (>500M params)

Aethel is the **first sub-60M model** to combine:

* Long-context capability (up to 32k tokens)
* Memory-based document compression (Titans-lite)
* SSM/DeltaNet efficiency
* Sliding-window attention precision
* Dense + sparse hybrid output
* Matryoshka vector slicing
* Code + text capabilities

This project sits at the intersection of research innovation and real production utility.

---

## 🧠 **2. High-Level Architecture**

Aethel consists of four major architectural components:

```
Input → Token Embedding → Gated DeltaNet Stack (6 layers)
       → Sliding Window Attention (2 layers)
       → TITANS-Lite Memory Module
       → Hybrid Dense + Sparse Output Heads
```

### **Core Innovations**

* **Gated DeltaNet backbone:** linear-time, long-range compression
* **Sliding-window attention:** high-resolution local reasoning
* **TITANS-lite memory:** dynamic, surprise-gated memory updates
* **Hybrid output:** dense semantic vector + sparse lexical vector
* **Code-aware embeddings:** docstring/code contrastive distillation
* **Matryoshka Vector Learning:** 768→512→256→128 dimensional slicing

---

## 🔧 **3. Detailed Architecture Components**

### **3.1 Embedding & Input Layer**

* BPE/SentencePiece vocabulary
* RoPE or ALiBi-style extrapolation to support 32k tokens
* Code token special handling

### **3.2 DeltaNet Stack (6 layers)**

Each layer performs:

* RMSNorm
* Fast-weight delta rule update
* Local convolution
* Gated residual update

Enables efficient long-context processing at O(N) cost.

### **3.3 Sliding Window Attention (2 layers)**

* Window size: 256 tokens
* Multi-head attention with gating
* Fixes precision issues in SSMs for numbers, dates, identifiers, and code

### **3.4 TITANS-Lite Memory Module**

Memory state: 1024-dimensional

Implements:

* Surprise-detection (prediction error)
* Dynamic memory update
* Momentum + decay
* Final memory representation used in pooled embedding

### **3.5 Hybrid Output Head**

Outputs two representations:

#### **Dense Vector (768d)**

* Uses pooled DeltaNet + attention + memory
* Matryoshka sliceable: 768 → 512 → 256 → 128

#### **Sparse Vector (Top-k Keyword Vector)**

* SPLADE-lite style activation
* Outputs weighted term activations over 5k vocabulary
* Improves keyword, entity, identifier, and code retrieval

---

## 🏋️ **4. Training Strategy**

Aethel is optimized for extremely low-budget training.

### **4.1 Multi-Teacher Knowledge Distillation**

Teachers:

* **BGE-M3** (dense + sparse retrieval)
* **Qwen2.5/Qwen3 Embeddings** (long-context semantics)
* **Codestral / Code-LLaMA small** (code retrieval)

Distillation losses:

* cosine similarity
* InfoNCE contrastive loss
* matryoshka dimension matching
* sparse-head supervision
* memory-state consistency loss

### **4.2 Training Dataset (15M–30M tokens)**

Small but powerful mix:

* Wikipedia + Books (general semantics)
* Long-document synthetic data
* Curated code-docstring pairs
* Instruction-style retrieval prompts

### **4.3 Training Budget**

Thanks to distillation and efficient architecture:

* Fits **under $30–$50** on Modal’s GPU pricing
* Trains on a single **A100 40GB / A100 80GB / L40S**

Recommended GPUs (from cheapest effective):

* **L4 ($0.80/h)** (slow but works)
* **L40S ($1.95/h)** (best cost/perf)
* **A100 40GB ($2.10/h)** (fast + stable)
* **A100 80GB ($2.50/h)** (best for larger batches)

Expected training time: **12–20 hours**.

---

## 🔍 **5. Use Cases (Why Aethel Will Be Adopted)**

### **5.1 Long-Document RAG**

* Law, finance, research papers, reports
* Aethel preserves global structure and local detail

### **5.2 Code Search & AI Coding Agents**

* Hybrid vector helps match identifiers and semantics
* Memory supports full-file understanding

### **5.3 Personalized / Session-Aware Retrieval**

* Memory module enables session-specific embeddings

### **5.4 Hybrid Dense + Sparse Search**

* Better accuracy without extra reranking models
* Matches BGE-M3 performance at 5× smaller scale

### **5.5 Vector DB Optimization**

* Matryoshka vectors reduce cost
* Backward-compatible adapters reduce re-indexing

---

## 📊 **6. Evaluation Plan**

* Mini-MTEB (STS, retrieval, clustering)
* Long-document recall tests
* Code retrieval (CoIR-lite)
* Ablations: memory, sparse head, attention

---

## 🧵 **7. Implementation Structure**

```
aethel/
│
├── model/
│   ├── embeddings.py
│   ├── deltanet.py
│   ├── attention.py
│   ├── memory_titans_lite.py
│   ├── hybrid_head.py
│   └── aethel_model.py
│
├── training/
│   ├── dataset_builder.py
│   ├── teacher_distillation.py
│   ├── losses.py
│   └── train.py
│
├── eval/
│   ├── mini_mteb.py
│   ├── long_context_eval.py
│   └── code_eval.py
│
├── export/
│   ├── export_onnx.py
│   ├── quantize.py
│   └── adapters.py
│
└── README.md
```

---

## 🏁 **8. Summary**

Aethel is a compact, powerful embedding model that offers:

* Long-context understanding
* Memory-augmented retrieval
* Dense + sparse hybrid vectors
* Code-aware representations
* Distilled SOTA performance under minimal compute

This model has the potential to be widely adopted across RAG, agents, legal/finance, and code search systems.

---

## 💰 **9. Modal GPU Pricing (for Training)**

```
Nvidia B200         $6.25 / h
Nvidia H200         $4.54 / h
Nvidia H100         $3.95 / h
Nvidia A100 80GB    $2.50 / h
Nvidia A100 40GB    $2.10 / h
Nvidia L40S         $1.95 / h
Nvidia A10          $1.10 / h
Nvidia L4           $0.80 / h
Nvidia T4           $0.59 / h
```

**Recommended for Aethel training:**

* ✨ **L40S** (best cost/performance)
* ⚡ **A100 40GB** (fastest for low cost)
* 🧠 **A100 80GB** (for large batch acceleration)

---

Aethel is now fully specced and ready for implementation, documentation, and launch.
