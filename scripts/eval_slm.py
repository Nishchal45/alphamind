"""Evaluate a claim-extraction model on a held-out set.

Runs the configured LLM backend over the validation JSONL and reports
claim-level precision / recall / F1 + the valid-JSON rate (ADR 0011).
The backend is whatever ``LLM_BACKEND`` selects, so the same script
scores:

- the fine-tuned model:   ``LLM_BACKEND=local SLM_ADAPTER_PATH=...``
- the untuned base model: ``LLM_BACKEND=local`` (no adapter)
- the frontier teacher:   ``LLM_BACKEND=anthropic`` (an upper bound)

Usage:

    LLM_BACKEND=local SLM_ADAPTER_PATH=checkpoints/claim-extractor/adapter \\
        uv run python scripts/eval_slm.py --val data/training/val.jsonl \\
            --out evals/slm_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
from pathlib import Path

from alphamind.config import get_settings
from alphamind.llm.factory import get_llm_client
from alphamind.training.dataset import read_jsonl
from alphamind.training.evaluate import DEFAULT_MATCH_THRESHOLD, evaluate_model

logger = logging.getLogger("alphamind.eval_slm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, required=True, help="val.jsonl path.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    return parser.parse_args()


def _configure_logging() -> None:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def _run(args: argparse.Namespace) -> int:
    _configure_logging()

    examples = read_jsonl(args.val)
    if not examples:
        print(f"No examples in {args.val}.", file=sys.stderr)
        return 1
    logger.info("evaluating over %d examples", len(examples))

    llm = get_llm_client()
    report = await evaluate_model(examples, llm=llm, threshold=args.threshold)

    print()
    print("Claim-extraction eval")
    print("---------------------")
    print(f"  examples         {report.n_examples}")
    print(f"  valid_json_rate  {report.valid_json_rate:.3f}")
    print(f"  precision        {report.precision:.3f}")
    print(f"  recall           {report.recall:.3f}")
    print(f"  f1               {report.f1:.3f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dataclasses.asdict(report), indent=2), encoding="utf-8")
        logger.info("wrote report to %s", args.out)

    return 0


def main() -> None:
    args = parse_args()
    try:
        code = asyncio.run(_run(args))
    finally:
        asyncio.run(_dispose())
    sys.exit(code)


async def _dispose() -> None:
    from alphamind.db.session import dispose_engine  # noqa: PLC0415

    await dispose_engine()


if __name__ == "__main__":
    main()
