from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer

from ..model.aethel_model import AethelModel


@dataclass
class LCFinetuneConfig:
    checkpoint_path: str
    save_path: str
    tokenizer_name: str = "BAAI/bge-base-en-v1.5"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    query_len: int = 128
    steps: int = 500
    lr: float = 5e-5
    chunk_tokens: int = 128
    sections: int = 8
    temperature: float = 0.05
    grad_accum: int = 1
    log_every: int = 50


def _build_doc(doc_id: int, sections: int, chunk_tokens: int) -> Tuple[List[str], str, int]:
    # Each section is a chunk; answer lives in section answer_idx.
    answer_idx = doc_id % sections
    chunks = []
    for s in range(sections):
        answer = f"value_{doc_id}_{s}"
        filler = " ".join([f"fill_{doc_id % 31}_{s}"] * (chunk_tokens // 4))
        chunk = f"doc{doc_id} section{s}: {answer} details. {filler}"
        chunks.append(chunk)
    query = f"What is in section {answer_idx} of doc {doc_id}?"
    return chunks, query, answer_idx


def run_long_context_finetune(cfg: LCFinetuneConfig):
    device = torch.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    model = AethelModel(vocab_size=32000, dim=768).to(device)
    ckpt = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    step = 0
    while step < cfg.steps:
        chunk_texts: List[str] = []
        query_texts: List[str] = []
        pos_indices: List[int] = []

        for b in range(cfg.batch_size):
            doc_chunks, query, answer_idx = _build_doc(doc_id=step * cfg.batch_size + b, sections=cfg.sections, chunk_tokens=cfg.chunk_tokens)
            base = len(chunk_texts)
            chunk_texts.extend(doc_chunks)
            query_texts.append(query)
            pos_indices.append(base + answer_idx)

        chunk_batch = tokenizer(
            chunk_texts,
            padding=True,
            truncation=True,
            max_length=cfg.chunk_tokens,
            return_tensors="pt",
        )
        query_batch = tokenizer(
            query_texts,
            padding=True,
            truncation=True,
            max_length=cfg.query_len,
            return_tensors="pt",
        )

        chunk_batch = {k: v.to(device) for k, v in chunk_batch.items()}
        query_batch = {k: v.to(device) for k, v in query_batch.items()}

        with autocast(enabled=torch.cuda.is_available()):
            chunk_emb = model(chunk_batch["input_ids"], mask=chunk_batch["attention_mask"]) ["dense"]
            query_emb = model(query_batch["input_ids"], mask=query_batch["attention_mask"]) ["dense"]
            chunk_emb = F.normalize(chunk_emb, dim=-1)
            query_emb = F.normalize(query_emb, dim=-1)
            sims = torch.matmul(query_emb, chunk_emb.T) / cfg.temperature
            labels = torch.tensor(pos_indices, device=device)
            loss = F.cross_entropy(sims, labels) / cfg.grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % cfg.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0:
            with torch.no_grad():
                topk = torch.topk(sims, k=min(3, sims.size(1)), dim=1).indices
                hits = (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()
            print(f"lc_step={step} loss={loss.item():.4f} hit@3={hits:.2f}")

        step += 1

    torch.save({"model": model.state_dict()}, cfg.save_path)
    print(f"Long-context finetune checkpoint saved to {cfg.save_path}")


def debug_overfit_one_batch(cfg: LCFinetuneConfig, steps: int = 50):
    """Sanity check: overfit a single synthetic batch and log memory stats."""
    device = torch.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    model = AethelModel(vocab_size=32000, dim=768).to(device)
    ckpt = torch.load(cfg.checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    if missing or unexpected:
        print(f"[load_warning] missing={missing} unexpected={unexpected}")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    # Build one batch up front.
    chunk_texts: List[str] = []
    query_texts: List[str] = []
    pos_indices: List[int] = []
    for b in range(cfg.batch_size):
        doc_chunks, query, answer_idx = _build_doc(doc_id=b, sections=cfg.sections, chunk_tokens=cfg.chunk_tokens)
        base = len(chunk_texts)
        chunk_texts.extend(doc_chunks)
        query_texts.append(query)
        pos_indices.append(base + answer_idx)

    chunk_batch = tokenizer(
        chunk_texts,
        padding=True,
        truncation=True,
        max_length=cfg.chunk_tokens,
        return_tensors="pt",
    )
    query_batch = tokenizer(
        query_texts,
        padding=True,
        truncation=True,
        max_length=cfg.query_len,
        return_tensors="pt",
    )
    chunk_batch = {k: v.to(device) for k, v in chunk_batch.items()}
    query_batch = {k: v.to(device) for k, v in query_batch.items()}
    labels = torch.tensor(pos_indices, device=device)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=torch.cuda.is_available()):
            chunk_emb = model(chunk_batch["input_ids"], mask=chunk_batch["attention_mask"], log_memory=(step % 10 == 0))["dense"]
            query_emb = model(query_batch["input_ids"], mask=query_batch["attention_mask"], log_memory=False)["dense"]
            chunk_emb = F.normalize(chunk_emb, dim=-1)
            query_emb = F.normalize(query_emb, dim=-1)
            sims = torch.matmul(query_emb, chunk_emb.T) / cfg.temperature
            loss = F.cross_entropy(sims, labels)
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            with torch.no_grad():
                topk = torch.topk(sims, k=min(3, sims.size(1)), dim=1).indices
                hits = (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()
            print(f"overfit_step={step} loss={loss.item():.4f} hit@3={hits:.2f}")

    print("Overfit sanity check done. If loss did not drop, memory/attention may be broken.")


__all__ = ["run_long_context_finetune", "LCFinetuneConfig", "debug_overfit_one_batch"]