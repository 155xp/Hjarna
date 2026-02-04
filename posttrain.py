import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict

import sentencepiece as spm
import torch

from data import build_loaders, infinite_loader
from model import BigramLanguageModel, ModelConfig

wandb_api_key = ""


@dataclass
class PostTrainConfig:
    dataset: str = "teknium/OpenHermes-2.5"
    text_field: str | None = None
    conversations_field: str | None = "conversations"
    cache_dir: str | None = None
    streaming: bool = True
    shuffle_buffer: int = 10_000
    val_examples: int = 5_000
    val_ratio: float = 0.001

    tokenizer_dir: str = "tokenizer"
    tokenizer_prefix: str = "owt2"
    eos_id: int = 2

    block_size: int = 2048
    batch_size: int = 16
    eval_batch_size: int = 4
    max_steps: int = 40_000
    eval_interval: int = 1_000
    eval_iters: int = 100
    max_data_gb: float = 5.0
    max_train_hours: float = 1.0

    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = -1
    min_lr: float = 0.0

    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0

    amp: bool = True
    bf16: bool | None = None

    num_workers: int = 2
    pin_memory: bool = True

    checkpoint_dir: str = "checkpoints/posttrain"
    save_interval: int = 1_000
    resume: str = ""
    log_interval: int = 10

    base_checkpoint: str = "checkpoints/pretrain/final.pt"
    seed: int = 1337


def parse_args() -> PostTrainConfig:
    parser = argparse.ArgumentParser(description="Post-train on OpenHermes 2.5")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-data-gb", type=float, default=None)
    parser.add_argument("--max-train-hours", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--base-checkpoint", default=None)

    args = parser.parse_args()
    cfg = PostTrainConfig()

    for key, value in vars(args).items():
        if value is None:
            continue
        if key == "no_streaming":
            cfg.streaming = False
        elif key == "no_amp":
            cfg.amp = False
        else:
            setattr(cfg, key, value)

    return cfg


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_schedule(step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr: float):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def save_checkpoint(path, model, optimizer, scaler, step, config, args):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "config": asdict(config),
        "args": asdict(args),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    step = ckpt.get("step", 0)
    if ckpt.get("rng_state") is not None:
        torch.set_rng_state(ckpt["rng_state"])
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    return step


def load_base_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    config = ModelConfig(**ckpt["config"])
    model = BigramLanguageModel(config).to(device)
    model.load_state_dict(ckpt["model"])
    return model, config


def load_tokenizer(args):
    model_prefix = os.path.join(args.tokenizer_dir, args.tokenizer_prefix)
    model_path = model_prefix + ".model"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Tokenizer not found. Run pretrain first to create it.")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.base_checkpoint):
        raise FileNotFoundError(f"Base checkpoint not found: {args.base_checkpoint}")

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    sp = load_tokenizer(args)

    model, config = load_base_checkpoint(args.base_checkpoint, device)
    args.block_size = config.block_size

    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
    )

    use_amp = args.amp and device.type == "cuda"
    if args.bf16 is None:
        use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    else:
        use_bf16 = args.bf16 and use_amp
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and not use_bf16)

    if args.warmup_steps < 0:
        warmup_steps = max(10, int(0.02 * args.max_steps))
    else:
        warmup_steps = args.warmup_steps

    wandb_run = None
    try:
        import wandb  # type: ignore
    except Exception:
        wandb = None

    if wandb:
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key
        if os.environ.get("WANDB_API_KEY"):
            wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "hjarna"),
                name=os.environ.get("WANDB_RUN_NAME"),
                config={"model": asdict(config), "posttrain": asdict(args)},
            )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": asdict(config), "args": asdict(args)}, f, indent=2)

    train_loader, val_loader = build_loaders(args, sp, build_val=True)
    train_iter = infinite_loader(train_loader)
    val_iter = infinite_loader(val_loader) if val_loader is not None else None

    start_step = 0
    if args.resume:
        resume_path = args.resume
        if args.resume == "latest":
            resume_path = os.path.join(args.checkpoint_dir, "last.pt")
        if os.path.exists(resume_path):
            start_step = load_checkpoint(resume_path, model, optimizer, scaler, device) + 1
            print(f"Resumed from {resume_path} at step {start_step}")
        else:
            print(f"Resume checkpoint not found: {resume_path}")

    def estimate_loss():
        if val_iter is None:
            return None
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(args.eval_iters):
                batch = next(val_iter)
                x = batch[:, :-1].to(device, non_blocking=True)
                y = batch[:, 1:].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    _, loss = model(x, y)
                losses.append(loss.item())
        model.train()
        return sum(losses) / max(1, len(losses))

    tokens_per_step = args.batch_size * args.block_size * args.grad_accum_steps
    last_log = time.time()
    start_time = time.time()

    model.train()
    final_step = start_step - 1
    for step in range(start_step, args.max_steps):
        final_step = step
        if args.max_train_hours > 0:
            elapsed = time.time() - start_time
            if elapsed >= args.max_train_hours * 3600:
                print(f"Reached time limit: {args.max_train_hours:.2f}h at step {step}")
                break

        lr = lr_schedule(step, warmup_steps, args.max_steps, args.learning_rate, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(args.grad_accum_steps):
            batch = next(train_iter)
            x = batch[:, :-1].to(device, non_blocking=True)
            y = batch[:, 1:].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps

            loss_accum += loss.item()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step % args.log_interval == 0:
            now = time.time()
            dt = now - last_log
            last_log = now
            tok_s = tokens_per_step / max(1e-6, dt)
            print(f"step {step} | loss {loss_accum:.4f} | lr {lr:.2e} | tok/s {tok_s:.0f}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": loss_accum,
                        "train/lr": lr,
                        "train/tokens_per_s": tok_s,
                    },
                    step=step,
                )

        if step % args.eval_interval == 0 and step != start_step:
            val_loss = estimate_loss()
            if val_loss is not None:
                print(f"eval step {step} | val loss {val_loss:.4f}")
                if wandb_run is not None:
                    wandb_run.log({"val/loss": val_loss}, step=step)

        if step % args.save_interval == 0 and step != start_step:
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step:07d}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scaler, step, config, args)
            last_path = os.path.join(args.checkpoint_dir, "last.pt")
            save_checkpoint(last_path, model, optimizer, scaler, step, config, args)

    if final_step < 0:
        final_step = 0
    final_path = os.path.join(args.checkpoint_dir, "final.pt")
    save_checkpoint(final_path, model, optimizer, scaler, final_step, config, args)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
