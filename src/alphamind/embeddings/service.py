"""Persistence layer for chunk embeddings.

Reads :class:`FilingChunk` rows whose embeddings are missing or stale
relative to the configured :class:`Embedder`, encodes them in batches,
and writes the resulting vectors back. A chunk is considered stale when
its ``embedding_model`` column does not match ``embedder.model_name``;
this lets the service handle backend swaps without manual intervention.

Two entry points mirror the chunking service:

- :func:`embed_chunks_for_document` — one filing document, expects an
  open session.
- :func:`embed_chunks_for_cik` — batch over every document for a CIK,
  opens its own session and isolates per-document failures so a single
  bad batch cannot abort the whole run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from alphamind.db.session import session_scope
from alphamind.embeddings.base import Embedder
from alphamind.models.company import Company
from alphamind.models.filing import Filing
from alphamind.models.filing_chunk import FilingChunk
from alphamind.models.filing_document import FilingDocument

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True, slots=True)
class EmbedIngestResult:
    """Outcome of embedding one filing document's chunks."""

    filing_document_id: int
    chunks_embedded: int
    chunks_skipped: int


@dataclass(frozen=True, slots=True)
class EmbedBatchResult:
    """Outcome of embedding every document's chunks for a CIK."""

    cik: str
    documents_seen: int
    documents_failed: int
    chunks_embedded: int
    chunks_skipped: int


async def embed_chunks_for_document(
    *,
    session: AsyncSession,
    embedder: Embedder,
    filing_document_id: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> EmbedIngestResult:
    """Embed chunks for one filing document.

    Skips chunks already embedded under ``embedder.model_name`` unless
    ``force`` is set. Vector dimension is validated against the embedder's
    declared dimension to catch model/schema mismatches early.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # Count any already-embedded chunks so the caller's skipped tally is
    # correct even when no new work happens.
    total_stmt = select(FilingChunk).where(FilingChunk.filing_document_id == filing_document_id)
    all_chunks = list((await session.execute(total_stmt)).scalars())

    if force:
        to_embed = list(all_chunks)
    else:
        to_embed = [
            c for c in all_chunks if c.embedding_model != embedder.model_name or c.embedding is None
        ]
    skipped = len(all_chunks) - len(to_embed)

    if not to_embed:
        return EmbedIngestResult(
            filing_document_id=filing_document_id,
            chunks_embedded=0,
            chunks_skipped=skipped,
        )

    embedded_count = 0
    for start in range(0, len(to_embed), batch_size):
        batch = to_embed[start : start + batch_size]
        vectors = await embedder.embed([c.text for c in batch])
        if len(vectors) != len(batch):
            raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(batch)} inputs")

        now = datetime.now(UTC)
        for chunk, vector in zip(batch, vectors, strict=True):
            if len(vector) != embedder.dimension:
                raise RuntimeError(
                    f"embedder {embedder.model_name!r} returned dim={len(vector)} "
                    f"but declared dim={embedder.dimension}"
                )
            chunk.embedding = vector
            chunk.embedding_model = embedder.model_name
            chunk.embedded_at = now

        # Flush each batch so a later failure inside the SAVEPOINT only loses
        # the in-flight batch, not earlier successes.
        await session.flush()
        embedded_count += len(batch)

    return EmbedIngestResult(
        filing_document_id=filing_document_id,
        chunks_embedded=embedded_count,
        chunks_skipped=skipped,
    )


async def embed_chunks_for_cik(
    cik: str,
    *,
    embedder: Embedder,
    form_types: frozenset[str] | None = None,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> EmbedBatchResult:
    """Embed chunks for every persisted filing document of a CIK.

    Iterates over documents (optionally filtered by form, optionally
    capped at ``limit``) and embeds each inside its own SAVEPOINT so a
    single failed batch leaves prior successes intact.
    """

    padded_cik = cik.strip().zfill(10)

    async with session_scope() as session:
        company_q = await session.execute(select(Company).where(Company.cik == padded_cik))
        company = company_q.scalar_one_or_none()
        if company is None:
            raise LookupError(
                f"company not ingested yet for cik={padded_cik!r}; run ingest_cik first"
            )

        docs_stmt = (
            select(FilingDocument)
            .join(Filing, Filing.id == FilingDocument.filing_id)
            .where(Filing.company_id == company.id)
        )
        if form_types is not None:
            docs_stmt = docs_stmt.where(Filing.form.in_(form_types))
        docs_stmt = docs_stmt.order_by(Filing.filing_date.desc())
        if limit is not None:
            docs_stmt = docs_stmt.limit(limit)

        documents = list((await session.execute(docs_stmt)).scalars())

        seen = len(documents)
        failed = 0
        total_embedded = 0
        total_skipped = 0

        for document in documents:
            try:
                async with session.begin_nested():
                    result = await embed_chunks_for_document(
                        session=session,
                        embedder=embedder,
                        filing_document_id=document.id,
                        batch_size=batch_size,
                        force=force,
                    )
            except Exception:
                logger.exception(
                    "embedding failed cik=%s filing_document_id=%s",
                    padded_cik,
                    document.id,
                )
                failed += 1
                continue

            total_embedded += result.chunks_embedded
            total_skipped += result.chunks_skipped

    logger.info(
        "embedding cik=%s seen=%d failed=%d embedded=%d skipped=%d model=%s",
        padded_cik,
        seen,
        failed,
        total_embedded,
        total_skipped,
        embedder.model_name,
    )

    return EmbedBatchResult(
        cik=padded_cik,
        documents_seen=seen,
        documents_failed=failed,
        chunks_embedded=total_embedded,
        chunks_skipped=total_skipped,
    )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "EmbedBatchResult",
    "EmbedIngestResult",
    "embed_chunks_for_cik",
    "embed_chunks_for_document",
]
