import argparse
import os
import random
import time

import sentencepiece as spm
import torch

from model import BigramLanguageModel, ModelConfig


def parse_args():
    parser = argparse.ArgumentParser(description="VRAM sanity check")
    parser.add_argument("--tokenizer-dir", default="tokenizer")
    parser.add_argument("--tokenizer-prefix", default="owt2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(tokenizer_dir, tokenizer_prefix):
    model_path = os.path.join(tokenizer_dir, tokenizer_prefix + ".model")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Tokenizer not found. Run pretrain once to create it.")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA not available.")

    set_seed(1337)
    torch.set_float32_matmul_precision("high")

    sp = load_tokenizer(args.tokenizer_dir, args.tokenizer_prefix)
    config = ModelConfig(
        vocab_size=sp.get_piece_size(),
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
        use_checkpointing=False,
    )

    model = BigramLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    use_amp = args.amp
    if args.bf16:
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    x = torch.randint(0, config.vocab_size, (args.batch_size, args.block_size), device=device)
    y = torch.randint(0, config.vocab_size, (args.batch_size, args.block_size), device=device)

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"steps={args.steps} | time={elapsed:.2f}s | peak_vram={peak:.2f} GB")


if __name__ == "__main__":
    main()
