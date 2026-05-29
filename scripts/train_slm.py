"""Fine-tune the claim-extraction SLM with QLoRA.

GPU-only. Reads the JSONL training set produced by
``scripts/build_training_set.py`` and runs the QLoRA fine-tune,
saving a LoRA adapter under ``--output-dir``. See ADR 0011 and
``docs/runbooks/fine-tune.md``.

Usage (on a CUDA box with the 'train' extra installed):

    uv sync --extra train
    uv run python scripts/train_slm.py \\
        --train data/training/train.jsonl \\
        --val   data/training/val.jsonl \\
        --output-dir checkpoints/claim-extractor
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from alphamind.training.dataset import read_jsonl
from alphamind.training.trainer import TrainConfig, TrainingDependencyError, train
from alphamind.training.types import DatasetSplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("alphamind.train_slm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="train.jsonl path.")
    parser.add_argument("--val", type=Path, default=None, help="Optional val.jsonl path.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/claim-extractor"))
    parser.add_argument("--base-model", default=TrainConfig().base_model)
    parser.add_argument("--epochs", type=float, default=TrainConfig().epochs)
    parser.add_argument("--lora-r", type=int, default=TrainConfig().lora_r)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig().learning_rate)
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantisation (needs much more VRAM).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_examples = read_jsonl(args.train)
    val_examples = read_jsonl(args.val) if args.val is not None else []
    if not train_examples:
        print(f"No training examples in {args.train}.", file=sys.stderr)
        sys.exit(1)

    split = DatasetSplit(train=tuple(train_examples), validation=tuple(val_examples))
    cfg = TrainConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lora_r=args.lora_r,
        learning_rate=args.learning_rate,
        load_in_4bit=not args.no_4bit,
    )

    try:
        adapter_path = train(split, cfg)
    except TrainingDependencyError as exc:
        print(f"{exc}", file=sys.stderr)
        sys.exit(2)

    logger.info("done. adapter at %s", adapter_path)
    print(f"\nTrained adapter saved to: {adapter_path}")
    print("Evaluate it with:")
    print(
        f"  LLM_BACKEND=local SLM_ADAPTER_PATH={adapter_path} "
        f"uv run python scripts/eval_slm.py --val {args.val or 'data/training/val.jsonl'}"
    )


if __name__ == "__main__":
    main()
