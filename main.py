import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd, env=None):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def parse_args():
    parser = argparse.ArgumentParser(description="Run pretrain then post-train.")
    parser.add_argument("--total-hours", type=float, default=None)
    parser.add_argument("--posttrain-ratio", type=float, default=0.25)
    parser.add_argument("--pretrain-hours", type=float, default=None)
    parser.add_argument("--posttrain-hours", type=float, default=None)
    parser.add_argument("--pretrain-data-gb", type=float, default=None)
    parser.add_argument("--posttrain-data-gb", type=float, default=None)
    parser.add_argument("--wandb-api-key", "--wanapi", dest="wandb_api_key", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    pretrain_dir = root / "checkpoints" / "pretrain"
    posttrain_dir = root / "checkpoints" / "posttrain"
    pretrain_ckpt = pretrain_dir / "final.pt"
    posttrain_ckpt = posttrain_dir / "final.pt"

    pretrain_hours = args.pretrain_hours
    posttrain_hours = args.posttrain_hours
    if args.total_hours is not None:
        ratio = args.posttrain_ratio
        ratio = max(0.0, min(1.0, ratio))
        posttrain_hours = args.total_hours * ratio
        pretrain_hours = args.total_hours - posttrain_hours

    run_env = None
    if args.wandb_api_key:
        run_env = os.environ.copy()
        run_env["WANDB_API_KEY"] = args.wandb_api_key

    if not pretrain_ckpt.exists():
        cmd = [
            sys.executable,
            str(root / "train.py"),
            "--checkpoint-dir",
            str(pretrain_dir),
        ]
        if pretrain_hours is not None:
            cmd += ["--max-train-hours", f"{pretrain_hours}"]
        if args.pretrain_data_gb is not None:
            cmd += ["--max-data-gb", f"{args.pretrain_data_gb}"]
        run([*cmd], env=run_env)
    else:
        print(f"Pretrain checkpoint exists: {pretrain_ckpt}")

    if not posttrain_ckpt.exists():
        cmd = [
            sys.executable,
            str(root / "posttrain.py"),
            "--base-checkpoint",
            str(pretrain_ckpt),
            "--checkpoint-dir",
            str(posttrain_dir),
        ]
        if posttrain_hours is not None:
            cmd += ["--max-train-hours", f"{posttrain_hours}"]
        if args.posttrain_data_gb is not None:
            cmd += ["--max-data-gb", f"{args.posttrain_data_gb}"]
        run([*cmd], env=run_env)
    else:
        print(f"Post-train checkpoint exists: {posttrain_ckpt}")


if __name__ == "__main__":
    main()
