from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dynamic_sandwich_dataset import DynamicSandwichDataset
from ..model.aethel_model import AethelModel


@dataclass
class SandwichConfig:
    checkpoint_path: str
    save_path: str
    tokenizer_name: str = "BAAI/bge-base-en-v1.5"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len: int = 8192
    batch_size: int = 4
    steps: int = 500
    lr: float = 5e-5
    temperature: float = 0.05
    grad_accum: int = 1
    log_every: int = 50
    num_workers: int = 0


def run_sandwich_finetune(cfg: SandwichConfig):
    device = torch.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    dataset = DynamicSandwichDataset(tokenizer=tokenizer, seq_len=cfg.seq_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    model = AethelModel(vocab_size=32000, dim=768).to(device)
    ckpt = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    step = 0
    for batch in loader:
        if step >= cfg.steps:
            break
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        query_batch = tokenizer(
            batch["query_text"],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        query_batch = {k: v.to(device) for k, v in query_batch.items()}

        with autocast(enabled=torch.cuda.is_available()):
            doc_out = model(input_ids, mask=attn)
            qry_out = model(query_batch["input_ids"], mask=query_batch.get("attention_mask"))
            doc_emb = F.normalize(doc_out["dense"], dim=-1)
            qry_emb = F.normalize(qry_out["dense"], dim=-1)
            sims = torch.matmul(qry_emb, doc_emb.T) / cfg.temperature
            labels = torch.arange(input_ids.size(0), device=device)
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
            print(f"sandwich_step={step} loss={loss.item():.4f} hit@3={hits:.2f}")

        step += 1

    torch.save({"model": model.state_dict()}, cfg.save_path)
    print(f"Sandwich finetune checkpoint saved to {cfg.save_path}")


__all__ = ["run_sandwich_finetune", "SandwichConfig"]
