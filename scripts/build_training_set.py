"""Build a distilled claim-extraction training set from ingested chunks.

Reads ``filing_chunks`` rows, runs each through the frontier model
(the teacher) to extract claims, and writes train / validation JSONL
under ``--out-dir``. See ADR 0011.

Usage:

    LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... \\
        uv run python scripts/build_training_set.py \\
            --limit 2000 --out-dir data/training

The teacher is whatever ``LLM_BACKEND`` selects. Run it against a real
frontier model (``anthropic``) to get usable targets; the ``echo``
default produces junk and exists only for wiring smoke tests.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from sqlalchemy import select

from alphamind.config import get_settings
from alphamind.db.session import dispose_engine, session_scope
from alphamind.llm.factory import get_llm_client
from alphamind.models.filing_chunk import FilingChunk
from alphamind.training.dataset import (
    ChunkInput,
    build_examples,
    split_examples,
    write_jsonl,
)

logger = logging.getLogger("alphamind.build_training_set")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2000, help="Max chunks to distil.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/training"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Drop chunks the teacher found no claims in (default: keep as negatives).",
    )
    return parser.parse_args()


def _configure_logging() -> None:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def _load_chunks(limit: int) -> list[ChunkInput]:
    async with session_scope() as session:
        stmt = select(FilingChunk.id, FilingChunk.text).order_by(FilingChunk.id).limit(limit)
        rows = (await session.execute(stmt)).all()
    return [ChunkInput(chunk_id=row.id, text=row.text) for row in rows]


async def _run(args: argparse.Namespace) -> int:
    _configure_logging()

    chunks = await _load_chunks(args.limit)
    if not chunks:
        print("No chunks found. Ingest + chunk some filings first.", file=sys.stderr)
        return 1
    logger.info("distilling %d chunks", len(chunks))

    llm = get_llm_client()
    examples, stats = await build_examples(chunks, llm=llm, keep_empty=not args.drop_empty)
    logger.info(
        "built %d examples (seen=%d unparseable=%d errored=%d)",
        stats.written,
        stats.seen,
        stats.dropped_unparseable,
        stats.dropped_errored,
    )
    if not examples:
        print("No usable examples produced.", file=sys.stderr)
        return 1

    split = split_examples(examples, validation_fraction=args.val_fraction, seed=args.seed)
    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val.jsonl"
    n_train = write_jsonl(split.train, train_path)
    n_val = write_jsonl(split.validation, val_path)
    logger.info("wrote %d train -> %s, %d val -> %s", n_train, train_path, n_val, val_path)
    return 0


def main() -> None:
    args = parse_args()
    try:
        code = asyncio.run(_run(args))
    finally:
        asyncio.run(dispose_engine())
    sys.exit(code)


if __name__ == "__main__":
    main()
