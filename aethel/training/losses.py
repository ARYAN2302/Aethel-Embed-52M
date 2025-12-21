from typing import Any, Dict

import torch
from torch.nn import functional as F


def cosine_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_n = F.normalize(x, dim=-1, eps=1e-8)
    y_n = F.normalize(y, dim=-1, eps=1e-8)
    return 1.0 - (x_n * y_n).sum(dim=-1).mean()


def sparse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    student_logprob = F.log_softmax(student_logits, dim=-1)
    teacher_prob = F.softmax(teacher_logits, dim=-1)
    return F.kl_div(student_logprob, teacher_prob, reduction="batchmean")


def matryoshka_loss(student_slices: Dict[int, torch.Tensor], teacher: torch.Tensor) -> torch.Tensor:
    losses = []
    for s, tensor in student_slices.items():
        losses.append(cosine_loss(tensor, teacher[..., :s]))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=teacher.device)


def info_nce_loss(query: torch.Tensor, key: torch.Tensor = None, temperature: float = 0.07) -> torch.Tensor:
    key = query if key is None else key
    min_dim = min(query.shape[-1], key.shape[-1])
    query = query[..., :min_dim]
    key = key[..., :min_dim]
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    logits = query @ key.t()
    logits = logits / max(temperature, 1e-6)
    labels = torch.arange(query.size(0), device=query.device)
    return F.cross_entropy(logits, labels)


def compute_losses(student_outputs: Dict[str, Any], teacher_outputs: Dict[str, Any], weights: Dict[str, float] = None, temperature: float = 0.07) -> Dict[str, torch.Tensor]:
    weights = weights or {
        "dense": 1.0,
        "infoNCE": 0.25,
        "matryoshka": 0.25,
        "sparse": 0.3,
        "memory": 0.2,
    }

    losses: Dict[str, torch.Tensor] = {}

    teacher_dense = None
    for key in ("dense", "sentence_embedding", "pooler_output"):
        if key in teacher_outputs:
            teacher_dense = teacher_outputs[key]
            break
    if teacher_dense is None and len(teacher_outputs) == 1:
        single_val = next(iter(teacher_outputs.values()))
        if isinstance(single_val, torch.Tensor):
            teacher_dense = single_val

    student_dense = student_outputs.get("dense")
    if teacher_dense is None:
        raise ValueError("Teacher dense outputs are required for distillation; received none.")

    if teacher_dense is not None and student_dense is not None:
        min_dim = min(student_dense.shape[-1], teacher_dense.shape[-1])
        losses["dense"] = weights["dense"] * cosine_loss(student_dense[..., :min_dim], teacher_dense[..., :min_dim])

    student_slices = student_outputs.get("dense_slices") or {}
    if teacher_dense is not None and student_slices:
        losses["matryoshka"] = weights["matryoshka"] * matryoshka_loss(student_slices, teacher_dense)

    teacher_sparse = teacher_outputs.get("sparse_logits")
    if teacher_sparse is None:
        teacher_sparse = teacher_outputs.get("sparse")
    student_sparse = student_outputs.get("sparse_logits")
    if teacher_sparse is not None and student_sparse is not None:
        if teacher_sparse.shape[-1] == student_sparse.shape[-1]:
            losses["sparse"] = weights["sparse"] * sparse_kl(student_sparse, teacher_sparse)

    student_memory = student_outputs.get("memory")
    if teacher_dense is not None and student_memory is not None:
        min_dim_mem = min(student_memory.shape[-1], teacher_dense.shape[-1])
        losses["memory"] = weights["memory"] * cosine_loss(student_memory[..., :min_dim_mem], teacher_dense[..., :min_dim_mem])

    if "infoNCE" in weights and student_dense is not None and teacher_dense is not None:
        losses["infoNCE"] = weights["infoNCE"] * info_nce_loss(student_dense, key=teacher_dense.detach(), temperature=temperature)

    if not losses:
        device = student_dense.device if student_dense is not None else next(iter(student_outputs.values())).device
        losses["dense"] = torch.tensor(0.0, device=device)

    cleaned = {}
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            cleaned[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            cleaned[k] = v

    return cleaned


__all__ = ["compute_losses", "cosine_loss", "sparse_kl", "matryoshka_loss", "info_nce_loss"]
