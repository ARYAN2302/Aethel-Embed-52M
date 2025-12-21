
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse

import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..model.aethel_model import AethelModel
from .dataset_builder import build_datasets, SplitSpec
from ..eval.mini_mteb import run_mini_mteb
from ..eval.long_context_eval import run_long_context_eval
from ..eval.code_eval import run_code_eval
from .teacher_distillation import distill_step
from .losses import compute_losses
from .teachers import load_teacher


@dataclass
class TrainConfig:
    vocab_size: int = 32000
    model_dim: int = 768
    seq_len: int = 1024
    batch_size: int = 16
    epochs: int = 3
    lr: float = 2e-4
    weight_decay: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps: int = 2000
    log_every: int = 20
    grad_accum: int = 1
    tokenizer_name: str = "BAAI/bge-base-en-v1.5"
    splits: List[Any] = field(
        default_factory=lambda: [
            SplitSpec(dataset_name="wikitext", config_name="wikitext-103-raw-v1", split="train[:20000]", text_field="text", weight=1.0, limit=2000),
            SplitSpec(dataset_name="ag_news", split="train[:20000]", text_field="text", weight=0.6, limit=2000),
            SplitSpec(dataset_name="amazon_polarity", split="train[:20000]", text_field="content", weight=0.4, limit=2000),
        ]
    )
    teacher_model_names: List[str] = field(default_factory=lambda: ["BAAI/bge-base-en-v1.5", "BAAI/bge-m3"])
    teacher_sparse: List[bool] = field(default_factory=lambda: [False, True])
    checkpoint_dir: str = "checkpoints"
    checkpoint_path: Optional[str] = None
    run_eval: bool = False
    debug_losses: bool = False
    checkpoint_steps: List[int] = field(default_factory=list)


def save_checkpoint(path: Path, model: Any, optimizer: optim.Optimizer, step: int, scheduler: Optional[Any] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def _build_config_from_args(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    cfg.device = args.device
    cfg.batch_size = args.batch_size
    cfg.max_steps = args.max_steps
    cfg.lr = args.lr
    cfg.seq_len = args.seq_len
    cfg.run_eval = args.run_eval
    cfg.debug_losses = args.debug_losses
    cfg.checkpoint_path = args.checkpoint_path
    return cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Aethel model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--debug-losses", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    return parser.parse_args()


def main(config: Optional[TrainConfig] = None):
    cfg = config or _build_config_from_args(_parse_args())
    device = torch.device(cfg.device)

    model = AethelModel(vocab_size=cfg.vocab_size, dim=cfg.model_dim).to(device)

    teachers: Dict[str, Any] = {}
    if len(cfg.teacher_sparse) < len(cfg.teacher_model_names):
        cfg.teacher_sparse = cfg.teacher_sparse + [False] * (len(cfg.teacher_model_names) - len(cfg.teacher_sparse))
    for name, sparse_flag in zip(cfg.teacher_model_names, cfg.teacher_sparse):
        teachers[name] = load_teacher(name, device=device, produce_sparse=sparse_flag)

    _, loader = build_datasets(cfg)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_steps, eta_min=cfg.lr * 0.1)
    step = 0
    checkpoint_set = set(cfg.checkpoint_steps or [])
    if cfg.checkpoint_path:
        ckpt = torch.load(cfg.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt.get("model", {}))
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"Warning: could not load scheduler state: {e}")
        step = int(ckpt.get("step", 0))
        if "scheduler" not in ckpt:
            scheduler.last_epoch = step - 1

    scaler = GradScaler(enabled=torch.cuda.is_available())
    model.train()
    epoch = 0
    while step < cfg.max_steps:
        for batch in loader:
            if step >= cfg.max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(enabled=torch.cuda.is_available()):
                need_ctx = cfg.debug_losses or step == 0
                result = distill_step(model, teachers, batch, compute_losses, return_context=need_ctx)
                if isinstance(result, tuple):
                    losses, ctx = result
                    if step == 0 or (cfg.debug_losses and step % cfg.log_every == 0):
                        print(
                            f"step={step} student_norm={ctx.get('student_dense_norm')} "
                            f"teacher_norm={ctx.get('teacher_dense_norm')} mask_sum={ctx.get('mask_sum')}"
                        )
                else:
                    losses = result
                total_loss = losses["total_loss"] / cfg.grad_accum
            scaler.scale(total_loss).backward(retain_graph=True)

            if (step + 1) % cfg.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if step % cfg.log_every == 0:
                loss_val = {k: v.item() if hasattr(v, "item") else v for k, v in losses.items()}
                lr = optimizer.param_groups[0]["lr"]
                print(f"step={step} epoch={epoch} lr={lr:.2e} loss={loss_val}")

            step += 1
        epoch += 1

        if step in checkpoint_set:
            ckpt_path = Path(cfg.checkpoint_dir) / f"aethel-step{step}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step, scheduler)
            print(f"Intermediate checkpoint saved to {ckpt_path}")

    ckpt_path = Path(cfg.checkpoint_dir) / f"aethel-step{step}.pt"
    save_checkpoint(ckpt_path, model, optimizer, step, scheduler)
    print(f"Checkpoint saved to {ckpt_path}")

    if cfg.run_eval:
        print("Running mini-MTEB (STS-lite)...")
        print(run_mini_mteb(model, tokenizer_name=cfg.tokenizer_name))
        print("Running synthetic long-context recall...")
        print(run_long_context_eval(model, tokenizer_name=cfg.tokenizer_name))
        print("Running code retrieval eval...")
        print(run_code_eval(model, tokenizer_name=cfg.tokenizer_name))


if __name__ == "__main__":
    main()


__all__ = ["main", "TrainConfig"]
