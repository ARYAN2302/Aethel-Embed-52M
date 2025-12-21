from typing import List, Optional

import modal

from .train import TrainConfig, main
# Alias the implementation to avoid clobbering by the Modal-decorated function name.
from .long_context_finetune import LCFinetuneConfig, run_long_context_finetune as run_lc_finetune_impl
from .sandwich_finetune import SandwichConfig, run_sandwich_finetune as run_sandwich_impl

image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")
volume = modal.Volume.from_name("aethel-checkpoints", create_if_missing=True)

app = modal.App("aethel-train")


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 2, volumes={"/vol": volume})
def run_remote(
    batch_size: int = 16,
    max_steps: int = 500,
    lr: float = 2e-4,
    seq_len: int = 1024,
    run_eval: bool = False,
    debug_losses: bool = False,
    checkpoint_path: str = None,
    checkpoint_steps: Optional[str] = None,
):
    cfg = TrainConfig()
    cfg.device = "cuda"
    cfg.batch_size = batch_size
    cfg.max_steps = max_steps
    cfg.lr = lr
    cfg.seq_len = seq_len
    cfg.run_eval = run_eval
    cfg.debug_losses = debug_losses
    cfg.checkpoint_dir = "/vol/checkpoints"
    if checkpoint_steps:
        cfg.checkpoint_steps = [int(x) for x in str(checkpoint_steps).split()] if isinstance(checkpoint_steps, str) else list(checkpoint_steps)
    else:
        cfg.checkpoint_steps = []
    if checkpoint_path:
        cfg.checkpoint_path = checkpoint_path
    main(cfg)


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 2, volumes={"/vol": volume})
def run_long_context_finetune(
    checkpoint_path: str,
    save_name: str = "aethel-step5000-lc1000.pt",
    steps: int = 500,
    lr: float = 5e-5,
    batch_size: int = 4,
    query_len: int = 128,
    chunk_tokens: int = 128,
    sections: int = 8,
    temperature: float = 0.05,
):
    save_path = f"/vol/checkpoints/{save_name}"
    cfg = LCFinetuneConfig(
        checkpoint_path=checkpoint_path,
        save_path=save_path,
        device="cuda",
        batch_size=batch_size,
        query_len=query_len,
        steps=steps,
        lr=lr,
        chunk_tokens=chunk_tokens,
        sections=sections,
        temperature=temperature,
    )
    run_lc_finetune_impl(cfg)


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 2, volumes={"/vol": volume})
def run_sandwich_finetune(
    checkpoint_path: str,
    save_name: str = "aethel-step5000-sandwich.pt",
    steps: int = 500,
    lr: float = 5e-5,
    batch_size: int = 4,
    seq_len: int = 8192,
    temperature: float = 0.05,
):
    save_path = f"/vol/checkpoints/{save_name}"
    cfg = SandwichConfig(
        checkpoint_path=checkpoint_path,
        save_path=save_path,
        device="cuda",
        seq_len=seq_len,
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        temperature=temperature,
    )
    run_sandwich_impl(cfg)


@app.local_entrypoint()
def run(
    device: str = "cpu",
    batch_size: int = 16,
    max_steps: int = 500,
    lr: float = 2e-4,
    seq_len: int = 1024,
    run_eval: bool = False,
    debug_losses: bool = False,
    checkpoint_path: str = None,
    checkpoint_steps: Optional[str] = None,
):
    cfg = TrainConfig()
    cfg.device = device
    cfg.batch_size = batch_size
    cfg.max_steps = max_steps
    cfg.lr = lr
    cfg.seq_len = seq_len
    cfg.run_eval = run_eval
    cfg.debug_losses = debug_losses
    if checkpoint_steps:
        cfg.checkpoint_steps = [int(x) for x in str(checkpoint_steps).split()] if isinstance(checkpoint_steps, str) else list(checkpoint_steps)
    else:
        cfg.checkpoint_steps = []
    if checkpoint_path:
        cfg.checkpoint_path = checkpoint_path
    main(cfg)
