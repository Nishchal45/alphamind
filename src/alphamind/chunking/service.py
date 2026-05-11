"""Persistence layer for the filing-body chunker.

Reads a filing document's bytes via the configured :class:`StorageBackend`,
runs the chunker, and writes the resulting :class:`FilingChunk` rows. The
service is idempotent: when chunks already exist for the document under
the same ``source_content_hash`` as the current body, the work is skipped.

Two entry points:

- :func:`chunk_filing_document` — single document, expects an open session.
- :func:`chunk_bodies_for_cik` — batch over every document for a CIK,
  opens its own session and isolates per-document failures so a single
  bad body cannot abort the whole run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from alphamind.chunking.chunker import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    chunk_filing,
)
from alphamind.db.session import session_scope
from alphamind.models.company import Company
from alphamind.models.filing import Filing
from alphamind.models.filing_chunk import FilingChunk
from alphamind.models.filing_document import FilingDocument
from alphamind.storage.base import StorageBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChunkIngestResult:
    """Outcome of chunking a single filing document."""

    filing_document_id: int
    chunks_written: int
    was_skipped: bool


@dataclass(frozen=True, slots=True)
class ChunkBatchResult:
    """Outcome of chunking every filing document for a CIK."""

    cik: str
    documents_seen: int
    documents_chunked: int
    documents_skipped: int
    documents_failed: int
    chunks_written: int


async def chunk_filing_document(
    *,
    storage: StorageBackend,
    session: AsyncSession,
    document: FilingDocument,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    force: bool = False,
) -> ChunkIngestResult:
    """Chunk one filing document and persist the chunks.

    When ``force`` is ``False`` (the default) and existing chunks already
    cover ``document.content_hash``, the body is not refetched and no rows
    are written. When the parent hash has changed (or ``force=True``), any
    stale chunks for the document are deleted and replaced with a freshly
    computed set.
    """

    if not force:
        existing_q = await session.execute(
            select(func.count())
            .select_from(FilingChunk)
            .where(
                FilingChunk.filing_document_id == document.id,
                FilingChunk.source_content_hash == document.content_hash,
            )
        )
        if existing_q.scalar_one() > 0:
            return ChunkIngestResult(
                filing_document_id=document.id,
                chunks_written=0,
                was_skipped=True,
            )

    body = await storage.get(document.storage_uri)
    chunks = chunk_filing(body, max_chars=max_chars, overlap_chars=overlap_chars)

    # Replace any prior chunks for this document. Cheaper and simpler than
    # diffing chunk-by-chunk; chunk counts are at most a few hundred.
    await session.execute(delete(FilingChunk).where(FilingChunk.filing_document_id == document.id))

    if chunks:
        rows = [
            {
                "filing_document_id": document.id,
                "section_label": c.section_label,
                "section_title": c.section_title,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "char_count": c.char_count,
                "source_content_hash": document.content_hash,
            }
            for c in chunks
        ]
        await session.execute(insert(FilingChunk).values(rows))

    return ChunkIngestResult(
        filing_document_id=document.id,
        chunks_written=len(chunks),
        was_skipped=False,
    )


async def chunk_bodies_for_cik(
    storage: StorageBackend,
    cik: str,
    *,
    form_types: frozenset[str] | None = None,
    limit: int | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    force: bool = False,
) -> ChunkBatchResult:
    """Chunk every persisted filing document for a CIK.

    Iterates over :class:`FilingDocument` rows attached to the company's
    filings (optionally filtered by form type, optionally capped at
    ``limit``). Each document is chunked inside its own SAVEPOINT so a
    single failure leaves prior successes intact.
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
        chunked = 0
        skipped = 0
        failed = 0
        total_chunks = 0

        for document in documents:
            try:
                async with session.begin_nested():
                    result = await chunk_filing_document(
                        storage=storage,
                        session=session,
                        document=document,
                        max_chars=max_chars,
                        overlap_chars=overlap_chars,
                        force=force,
                    )
            except Exception:
                logger.exception(
                    "chunking failed cik=%s filing_document_id=%s",
                    padded_cik,
                    document.id,
                )
                failed += 1
                continue

            if result.was_skipped:
                skipped += 1
            else:
                chunked += 1
                total_chunks += result.chunks_written

    logger.info(
        "chunking cik=%s seen=%d chunked=%d skipped=%d failed=%d chunks=%d",
        padded_cik,
        seen,
        chunked,
        skipped,
        failed,
        total_chunks,
    )

    return ChunkBatchResult(
        cik=padded_cik,
        documents_seen=seen,
        documents_chunked=chunked,
        documents_skipped=skipped,
        documents_failed=failed,
        chunks_written=total_chunks,
    )


__all__ = [
    "ChunkBatchResult",
    "ChunkIngestResult",
    "chunk_bodies_for_cik",
    "chunk_filing_document",
]
