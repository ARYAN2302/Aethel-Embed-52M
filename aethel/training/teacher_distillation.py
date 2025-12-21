from typing import Any, Dict, Optional

import torch


def run_teachers(teachers: Dict[str, Any], batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    for name, teacher in (teachers or {}).items():
        if teacher is None:
            continue
        encode = teacher.get("encode") if isinstance(teacher, dict) else getattr(teacher, "encode", None)
        if encode is None:
            raise ValueError(f"Teacher {name} missing encode callable")
        with torch.no_grad():
            out = encode(input_ids=input_ids, attention_mask=attention_mask)
        outputs[name] = out
    return outputs


def _aggregate_teacher_outputs(teacher_outputs: Dict[str, Any]) -> Dict[str, Any]:
    dense_list = []
    sparse_list = []
    for out in teacher_outputs.values():
        if out is None:
            continue
        if "dense" in out:
            dense_list.append(out["dense"])
        if "sparse_logits" in out:
            sparse_list.append(out["sparse_logits"])
    aggregated: Dict[str, Any] = {}
    if dense_list:
        min_dim = min(t.shape[-1] for t in dense_list)
        aligned = [t[..., :min_dim] for t in dense_list]
        aggregated["dense"] = torch.stack(aligned).mean(dim=0)
    if sparse_list:
        min_vocab = min(t.shape[-1] for t in sparse_list)
        aligned_sparse = [t[..., :min_vocab] for t in sparse_list]
        aggregated["sparse_logits"] = torch.stack(aligned_sparse).mean(dim=0)
    return aggregated


def distill_step(student: Any, teachers: Dict[str, Any], batch: Dict[str, torch.Tensor], loss_fn: Any, optimizer: Optional[torch.optim.Optimizer] = None, detach: bool = False, return_context: bool = False):
    student_outputs = student(batch["input_ids"], mask=batch.get("attention_mask"))
    
    for k, v in list(student_outputs.items()):
        if isinstance(v, torch.Tensor):
            student_outputs[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        elif isinstance(v, dict):
            student_outputs[k] = {kk: torch.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
    
    per_teacher_outputs = run_teachers(teachers, batch)
    teacher_outputs = _aggregate_teacher_outputs(per_teacher_outputs)
    losses = loss_fn(student_outputs, teacher_outputs)
    
    ctx = None
    if return_context:
        sd = student_outputs.get("dense")
        td = teacher_outputs.get("dense")
        ctx = {
            "student_dense_norm": float(sd.norm().item()) if sd is not None else None,
            "teacher_dense_norm": float(td.norm().item()) if td is not None else None,
            "mask_sum": int(batch.get("attention_mask", torch.tensor([])).sum().item()) if batch.get("attention_mask") is not None else None,
        }
    
    total_loss = sum(losses.values()) if isinstance(losses, dict) else losses
    if optimizer is not None:
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if detach:
        if isinstance(losses, dict):
            out = {"total_loss": total_loss.detach(), **{k: v.detach() if hasattr(v, "detach") else v for k, v in losses.items()}}
        else:
            out = {"total_loss": total_loss.detach()}
    else:
        out = {"total_loss": total_loss, **losses} if isinstance(losses, dict) else {"total_loss": total_loss}

    if return_context:
        return out, (ctx or {})
    return out


__all__ = ["distill_step", "run_teachers"]
