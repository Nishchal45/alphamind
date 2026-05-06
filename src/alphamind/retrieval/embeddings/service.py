"""DB-aware embedding service.

Reads chunks for a filing, runs them through the embedder in batches,
writes vectors back to ``filing_chunks.embedding``. Idempotent: chunks
that already have a non-null embedding are skipped (set ``force=True`` to
re-embed everything).

Batching matters because real embedders amortise model load and GPU work
across the batch — a 32-item batch is much more than 32x faster than 32
single-item calls. The deterministic stub doesn't care, but the same code
path will be used by the real model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from alphamind.models.filing_chunk import FilingChunk
from alphamind.retrieval.embeddings.base import Embedder

logger = logging.getLogger(__name__)


DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Outcome of running the embedder over one filing's chunks."""

    filing_id: int
    chunks_seen: int
    chunks_embedded: int
    chunks_skipped: int


async def embed_chunks_for_filing(
    *,
    embedder: Embedder,
    session: AsyncSession,
    filing_id: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> EmbeddingResult:
    """Embed every chunk belonging to ``filing_id``.

    Parameters
    ----------
    force:
        If ``True``, re-embed even chunks that already have a vector. Use
        this after switching embedder backends.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    stmt = select(FilingChunk).where(FilingChunk.filing_id == filing_id)
    if not force:
        stmt = stmt.where(FilingChunk.embedding.is_(None))
    stmt = stmt.order_by(FilingChunk.ordinal)

    chunks = list((await session.execute(stmt)).scalars())

    seen_q = await session.execute(select(FilingChunk.id).where(FilingChunk.filing_id == filing_id))
    total_chunks = len(list(seen_q.scalars()))

    if not chunks:
        return EmbeddingResult(
            filing_id=filing_id,
            chunks_seen=total_chunks,
            chunks_embedded=0,
            chunks_skipped=total_chunks,
        )

    embedded = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = await embedder.embed([c.text for c in batch])
        if len(vectors) != len(batch):
            raise ValueError(f"embedder returned {len(vectors)} vectors for {len(batch)} inputs")
        for chunk, vector in zip(batch, vectors, strict=True):
            if len(vector) != embedder.dim:
                raise ValueError(
                    f"embedder dim mismatch: got {len(vector)}, expected {embedder.dim}"
                )
            await session.execute(
                update(FilingChunk).where(FilingChunk.id == chunk.id).values(embedding=vector)
            )
            embedded += 1

    logger.info(
        "embedded chunks filing_id=%d seen=%d embedded=%d",
        filing_id,
        total_chunks,
        embedded,
    )

    return EmbeddingResult(
        filing_id=filing_id,
        chunks_seen=total_chunks,
        chunks_embedded=embedded,
        chunks_skipped=total_chunks - embedded,
    )
