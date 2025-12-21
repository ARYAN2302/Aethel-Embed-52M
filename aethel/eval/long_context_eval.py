from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


def _make_synthetic(num_docs: int = 8, sections: int = 8) -> Dict[str, str]:
    docs = []
    queries = []
    answers = []
    for d in range(num_docs):
        doc_parts = []
        for s in range(sections):
            part = f"doc{d} section{s} : value_{d}_{s} and extra context."
            doc_parts.append(part)
        doc = " \n".join(doc_parts)
        answer_idx = d % sections
        answer = f"value_{d}_{answer_idx}"
        query = f"What is in section {answer_idx} of doc {d}?"
        docs.append(doc)
        queries.append(query)
        answers.append(answer)
    return {"docs": docs, "queries": queries, "answers": answers}


def _chunk_tokens(tokenizer, text: str, chunk_tokens: int) -> List[List[int]]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [ids[i : i + chunk_tokens] for i in range(0, len(ids), chunk_tokens) if ids[i : i + chunk_tokens]]


def _encode_long_context(model: Any, tokenizer, text: str, chunk_tokens: int, device) -> Tuple[torch.Tensor, List[str]]:
    """Encode a long document chunk-by-chunk while carrying memory forward."""
    chunks = _chunk_tokens(tokenizer, text, chunk_tokens)
    memory_state = None
    embeds = []
    chunk_texts: List[str] = []
    for chunk_ids in chunks:
        enc = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        input_ids = torch.tensor(chunk_ids, device=device).unsqueeze(0)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = model.forward_with_memory(input_ids, mask=attn, memory_state=memory_state)
        memory_state = outputs.get("memory")
        embeds.append(outputs["dense"])
        chunk_texts.append(enc)
    if len(embeds) == 0:
        return torch.empty(0, device=device), []
    return torch.cat(embeds, dim=0), chunk_texts


def _encode_query(model: Any, tokenizer, query: str, max_len: int, device) -> torch.Tensor:
    batch = tokenizer([query], return_tensors="pt", truncation=True, max_length=max_len)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model.forward_with_memory(batch["input_ids"], mask=batch.get("attention_mask"), memory_state=None)
    return outputs["dense"]


def _normalize_text(s: str) -> str:
    return s.replace(" ", "").lower()


def run_long_context_eval(model: Any, tokenizer_name: str = "BAAI/bge-base-en-v1.5", chunk_tokens: int = 128, k: int = 5) -> Dict[str, float]:
    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = _make_synthetic()

    hits = 0
    total = len(data["docs"])
    for doc, query, answer in zip(data["docs"], data["queries"], data["answers"]):
        chunk_embs, chunk_texts = _encode_long_context(model, tokenizer, doc, chunk_tokens, device)
        query_emb = _encode_query(model, tokenizer, query, max_len=chunk_tokens, device=device)
        if chunk_embs.numel() == 0:
            continue
        sims = _cosine(query_emb, chunk_embs).view(-1)
        k_eff = min(k, sims.numel())
        topk_idx = torch.topk(sims, k=k_eff).indices.view(-1)
        topk_chunks = [chunk_texts[int(i.item())] for i in topk_idx]
        ans_norm = _normalize_text(answer)
        if any(ans_norm in _normalize_text(ch) for ch in topk_chunks):
            hits += 1

    recall_atk = hits / total
    return {f"long_context_recall@{k}": recall_atk}


__all__ = ["run_long_context_eval"]
