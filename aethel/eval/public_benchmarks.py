"""Public benchmark helpers: NQ long-context, GovReport chunk matching, and a small MTEB-style suite."""

from typing import Any, Dict, Iterable, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


def _chunk_tokens(tokenizer, text: str, chunk_tokens: int) -> List[List[int]]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [ids[i : i + chunk_tokens] for i in range(0, len(ids), chunk_tokens) if ids[i : i + chunk_tokens]]


def _encode_chunks(model: Any, tokenizer, text: str, chunk_tokens: int, device) -> Tuple[torch.Tensor, List[str]]:
    chunks = _chunk_tokens(tokenizer, text, chunk_tokens)
    memory_state = None
    chunk_embs: List[torch.Tensor] = []
    chunk_texts: List[str] = []
    for chunk_ids in chunks:
        decoded = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        input_ids = torch.tensor(chunk_ids, device=device).unsqueeze(0)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = model.forward_with_memory(input_ids, mask=attn, memory_state=memory_state)
        memory_state = outputs.get("memory")
        chunk_embs.append(outputs["dense"])
        chunk_texts.append(decoded)
    if len(chunk_embs) == 0:
        return torch.empty(0, device=device), []
    return torch.cat(chunk_embs, dim=0), chunk_texts


def _encode_query(model: Any, tokenizer, query: str, max_len: int, device) -> torch.Tensor:
    batch = tokenizer([query], return_tensors="pt", truncation=True, max_length=max_len)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model.forward_with_memory(batch["input_ids"], mask=batch.get("attention_mask"), memory_state=None)
    return outputs["dense"]


def _answer_in_chunks(answer: str, chunks: Iterable[str]) -> bool:
    norm_ans = answer.replace(" ", "").lower()
    for ch in chunks:
        if norm_ans in ch.replace(" ", "").lower():
            return True
    return False


def run_nq_long_context(
    model: Any,
    tokenizer_name: str = "BAAI/bge-base-en-v1.5",
    chunk_tokens: int = 256,
    k: int = 5,
    limit: int = 32,
) -> Dict[str, float]:
    """Recall@k on Natural Questions document slices.

    Uses the long document text; success if any top-k chunk contains the gold short answer string.
    """
    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds_stream = load_dataset("natural_questions", split="train", streaming=True)
    ds = list(iter(ds_stream.take(limit)))

    hits = 0
    total = 0
    for ex in ds:
        doc_text = ex.get("document_text") or ""
        answers = ex.get("annotations", [{}])
        # Extract first short answer string if present.
        answer = None
        for ann in answers:
            short_ans = ann.get("short_answers") if isinstance(ann, dict) else None
            if short_ans:
                # short_answers is a list of dicts with start/end and text tokens
                tokens = ann.get("tokens") or ex.get("document_tokens") or []
                if tokens and isinstance(tokens, list):
                    start = short_ans[0].get("start_token")
                    end = short_ans[0].get("end_token")
                    if start is not None and end is not None:
                        text_slice = " ".join(tokens[start:end])
                        answer = text_slice.strip()
                        break
                # Fallback: try provided text field
                if short_ans[0].get("text"):
                    answer = short_ans[0]["text"]
                    break
        if not doc_text or not answer:
            continue
        total += 1
        chunk_embs, chunk_texts = _encode_chunks(model, tokenizer, doc_text, chunk_tokens, device)
        query_emb = _encode_query(model, tokenizer, ex.get("question_text", ""), max_len=chunk_tokens, device=device)
        if chunk_embs.numel() == 0:
            continue
        sims = _cosine(query_emb, chunk_embs).view(-1)
        k_eff = min(k, sims.numel())
        top_idx = torch.topk(sims, k=k_eff).indices
        top_chunks = [chunk_texts[int(i)] for i in top_idx]
        if _answer_in_chunks(str(answer), top_chunks):
            hits += 1
    return {"nq_recall@5": hits / max(total, 1)}


def run_govreport_matching(
    model: Any,
    tokenizer_name: str = "BAAI/bge-base-en-v1.5",
    chunk_tokens: int = 256,
    k: int = 5,
    limit: int = 16,
    queries_per_doc: int = 3,
) -> Dict[str, float]:
    """Recall@1/5 on GovReport summary-to-chunk retrieval."""
    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    try:
        ds_stream = load_dataset("ccdv/govreport", split="train", streaming=True)
    except Exception:
        ds_stream = load_dataset("ccdv/govreport-summarization", split="train", streaming=True)
    ds = list(iter(ds_stream.take(limit)))

    hits1 = 0
    hits5 = 0
    total = 0
    for ex in ds:
        article = ex.get("article")
        summary = ex.get("summary")
        if not article or not summary:
            continue
        article_text = "\n".join(article if isinstance(article, list) else [article])
        summary_sents = summary if isinstance(summary, list) else [summary]
        queries = summary_sents[:queries_per_doc]
        chunk_embs, chunk_texts = _encode_chunks(model, tokenizer, article_text, chunk_tokens, device)
        if chunk_embs.numel() == 0:
            continue
        for q in queries:
            total += 1
            q_emb = _encode_query(model, tokenizer, q, max_len=chunk_tokens, device=device)
            sims = _cosine(q_emb, chunk_embs).view(-1)
            topk = torch.topk(sims, k=min(k, sims.numel())).indices
            top1 = int(torch.topk(sims, k=1).indices[0])
            top_chunks = [chunk_texts[int(i)] for i in topk]
            if q.strip() and q.strip().lower() in chunk_texts[top1].lower():
                hits1 += 1
            if any(q.strip().lower() in ch.lower() for ch in top_chunks):
                hits5 += 1
    denom = max(total, 1)
    return {"govreport_recall@1": hits1 / denom, "govreport_recall@5": hits5 / denom}


def run_small_mteb_suite(
    model: Any,
    tokenizer_name: str = "BAAI/bge-base-en-v1.5",
    limit: int = 500,
) -> Dict[str, float]:
    """Lightweight generalization suite over common public datasets."""
    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    metrics: Dict[str, float] = {}

    # STS-B (dev subset)
    stsb_stream = load_dataset("stsb_multi_mt", name="en", split="dev", streaming=True)
    stsb_samples = list(iter(stsb_stream.take(limit)))
    s1 = tokenizer([ex["sentence1"] for ex in stsb_samples], padding=True, truncation=True, max_length=256, return_tensors="pt")
    s2 = tokenizer([ex["sentence2"] for ex in stsb_samples], padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        e1 = model(s1["input_ids"].to(device), mask=s1["attention_mask"].to(device))["dense"]
        e2 = model(s2["input_ids"].to(device), mask=s2["attention_mask"].to(device))["dense"]
    sims = _cosine(e1, e2).cpu()
    labels = torch.tensor([ex["similarity_score"] for ex in stsb_samples]) / 5.0
    vx = sims - sims.mean()
    vy = labels - labels.mean()
    denom = torch.sqrt((vx ** 2).sum() * (vy ** 2).sum()).clamp(min=1e-9)
    metrics["stsb_pearson"] = (vx * vy).sum().item() / denom.item()

    # QQP duplicate detection via cosine threshold
    qqp_stream = load_dataset("glue", "qqp", split="validation", streaming=True)
    qqp_samples = list(iter(qqp_stream.take(limit)))
    q1 = tokenizer([ex["question1"] for ex in qqp_samples], padding=True, truncation=True, max_length=128, return_tensors="pt")
    q2 = tokenizer([ex["question2"] for ex in qqp_samples], padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        e1 = model(q1["input_ids"].to(device), mask=q1["attention_mask"].to(device))["dense"]
        e2 = model(q2["input_ids"].to(device), mask=q2["attention_mask"].to(device))["dense"]
    cos = _cosine(e1, e2).cpu()
    preds = (cos > 0.5).int()
    labels = torch.tensor([ex["label"] for ex in qqp_samples])
    metrics["qqp_acc"] = (preds == labels).float().mean().item()

    # Amazon Polarity: nearest-centroid unsupervised classification
    amazon_stream = load_dataset("amazon_polarity", split="train", streaming=True)
    amazon_samples = list(iter(amazon_stream.take(limit)))
    texts = [f"{ex['title']}\n{ex['content']}" for ex in amazon_samples]
    batch = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        emb = model(batch["input_ids"].to(device), mask=batch["attention_mask"].to(device))["dense"]
    labels = torch.tensor([ex["label"] for ex in amazon_samples]).to(device)
    centroids = []
    for cls in [0, 1]:
        mask = labels == cls
        centroids.append(torch.nn.functional.normalize(emb[mask].mean(dim=0, keepdim=True), dim=-1))
    sims = torch.stack([_cosine(emb, c) for c in centroids], dim=-1)  # (N, 2)
    preds = sims.argmax(dim=-1)
    metrics["amazon_polarity_acc"] = (preds == labels).float().mean().item()

    # AG News: nearest-centroid over 4 topics
    ag_stream = load_dataset("ag_news", split="train", streaming=True)
    ag_samples = list(iter(ag_stream.take(limit)))
    ag_texts = []
    ag_labels_list = []
    for ex in ag_samples:
        title = ex.get("title") or ""
        desc = ex.get("description") or ex.get("text") or ""
        ag_texts.append(f"{title}\n{desc}")
        ag_labels_list.append(ex.get("label", 0))
    ag_batch = tokenizer(ag_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        ag_emb = model(ag_batch["input_ids"].to(device), mask=ag_batch["attention_mask"].to(device))["dense"]
    ag_labels = torch.tensor(ag_labels_list).to(device)
    ag_centroids = []
    for cls in range(4):
        mask = ag_labels == cls
        ag_centroids.append(torch.nn.functional.normalize(ag_emb[mask].mean(dim=0, keepdim=True), dim=-1))
    ag_sims = torch.stack([_cosine(ag_emb, c) for c in ag_centroids], dim=-1)
    ag_preds = ag_sims.argmax(dim=-1)
    metrics["ag_news_acc"] = (ag_preds == ag_labels).float().mean().item()

    return metrics


__all__ = [
    "run_nq_long_context",
    "run_govreport_matching",
    "run_small_mteb_suite",
]
