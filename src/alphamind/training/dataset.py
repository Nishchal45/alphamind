"""Distillation dataset builder + JSONL IO + train/val split.

The builder turns filing chunks into ``(chunk, claims)`` training
examples by running each chunk through the frontier model (the
``LLMClient`` Protocol, ADR 0006). Because it depends only on the
Protocol and an iterable of ``(chunk_id, text)`` pairs, it's testable
end-to-end with a scripted stub — no API call, no database, in the
test suite. The DB read that feeds it in production is a thin shell in
``scripts/build_training_set.py``.

A teacher output that doesn't parse as the expected schema is dropped
(a teacher mistake shouldn't become a training target) and counted, so
the caller can see the teacher's malformed-output rate.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from alphamind.llm.base import LLMClient, LLMClientError
from alphamind.training.formatting import (
    build_extraction_messages,
    claims_to_json,
    parse_claims,
)
from alphamind.training.types import DatasetSplit, TrainingExample

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BuildStats:
    """Bookkeeping from a dataset build."""

    seen: int
    written: int
    dropped_unparseable: int
    dropped_errored: int


@dataclass(frozen=True, slots=True)
class ChunkInput:
    """One source chunk for the builder. Decoupled from the ORM row."""

    chunk_id: int
    text: str


async def build_examples(
    chunks: Sequence[ChunkInput],
    *,
    llm: LLMClient,
    model: str | None = None,
    max_tokens: int = 1024,
    keep_empty: bool = True,
) -> tuple[list[TrainingExample], BuildStats]:
    """Distil ``chunks`` into training examples via the teacher ``llm``.

    Parameters
    ----------
    keep_empty:
        When True (default), a chunk the teacher found no claims in
        still becomes an example with an empty claim list — useful
        negative signal. When False, those are dropped.
    """
    examples: list[TrainingExample] = []
    dropped_unparseable = 0
    dropped_errored = 0

    for chunk in chunks:
        messages = build_extraction_messages(chunk.text)
        try:
            response = await llm.complete(messages, model=model, max_tokens=max_tokens)
        except LLMClientError as exc:
            logger.warning("chunk %s: teacher call failed: %s", chunk.chunk_id, exc)
            dropped_errored += 1
            continue

        claims = parse_claims(response.content)
        if claims is None:
            logger.debug("chunk %s: teacher output unparseable, dropping", chunk.chunk_id)
            dropped_unparseable += 1
            continue

        if not claims and not keep_empty:
            continue

        examples.append(
            TrainingExample(
                chunk_id=chunk.chunk_id,
                chunk_text=chunk.text,
                claims=tuple(claims),
            )
        )

    stats = BuildStats(
        seen=len(chunks),
        written=len(examples),
        dropped_unparseable=dropped_unparseable,
        dropped_errored=dropped_errored,
    )
    return examples, stats


def write_jsonl(examples: Iterable[TrainingExample], path: Path) -> int:
    """Write examples to JSONL. Returns the number of rows written.

    Each line is ``{"chunk_id": int, "chunk_text": str, "target": str}``
    where ``target`` is the canonical claims-JSON string the model is
    trained to emit. Keeping the target as the exact training string
    (not a nested object) means the trainer reads it verbatim.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            row = {
                "chunk_id": ex.chunk_id,
                "chunk_text": ex.chunk_text,
                "target": claims_to_json(ex.claims),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[TrainingExample]:
    """Read examples back from a JSONL file written by :func:`write_jsonl`.

    Rows whose ``target`` doesn't parse are skipped with a warning —
    a corrupted line shouldn't abort a training run, but it also
    shouldn't silently become an empty target.
    """
    examples: list[TrainingExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            claims = parse_claims(row["target"])
            if claims is None:
                logger.warning("%s:%d: unparseable target, skipping", path, line_no)
                continue
            examples.append(
                TrainingExample(
                    chunk_id=int(row["chunk_id"]),
                    chunk_text=str(row["chunk_text"]),
                    claims=tuple(claims),
                )
            )
    return examples


def _split_hash(chunk_id: int, seed: int) -> float:
    """Deterministic [0, 1) hash of a chunk id, seeded.

    Hashing the id (rather than shuffling positions) makes the split
    stable across dataset rebuilds: a chunk lands in the same partition
    every time, so adding new chunks doesn't reshuffle the old ones
    between train and validation.
    """
    digest = hashlib.sha256(f"{seed}:{chunk_id}".encode()).hexdigest()
    # First 8 hex digits → [0, 1).
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_examples(
    examples: Sequence[TrainingExample],
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> DatasetSplit:
    """Partition examples into train / validation by hashed chunk id."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")

    train: list[TrainingExample] = []
    validation: list[TrainingExample] = []
    for ex in examples:
        if _split_hash(ex.chunk_id, seed) < validation_fraction:
            validation.append(ex)
        else:
            train.append(ex)
    return DatasetSplit(train=tuple(train), validation=tuple(validation))


__all__ = [
    "BuildStats",
    "ChunkInput",
    "build_examples",
    "read_jsonl",
    "split_examples",
    "write_jsonl",
]
