"""DB-aware chunking service.

Reads a filing's stored body via :class:`alphamind.storage.StorageBackend`,
runs it through :class:`ChunkingPipeline`, and upserts the resulting chunks
into ``filing_chunks``. The natural key is ``(filing_id, ordinal)`` —
re-chunking the same filing replaces the old chunk set atomically.

The atomicity matters: if we deleted-then-inserted in two steps and the
process died between, retrieval would briefly see a half-chunked filing.
We do delete-then-insert *inside one transaction* for that reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from alphamind.db.session import session_scope
from alphamind.models.company import Company
from alphamind.models.filing import Filing
from alphamind.models.filing_chunk import FilingChunk
from alphamind.models.filing_document import FilingDocument
from alphamind.retrieval.chunking.base import Chunk, ChunkingConfig
from alphamind.retrieval.chunking.pipeline import ChunkingPipeline
from alphamind.storage.base import StorageBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    """Outcome of chunking one filing."""

    filing_id: int
    chunks_written: int
    chunks_replaced: int
    skipped_reason: str | None = None


async def _load_chunks(
    session: AsyncSession,
    filing_id: int,
) -> list[FilingChunk]:
    result = await session.execute(select(FilingChunk).where(FilingChunk.filing_id == filing_id))
    return list(result.scalars())


async def chunk_filing(
    *,
    storage: StorageBackend,
    session: AsyncSession,
    filing_id: int,
    pipeline: ChunkingPipeline | None = None,
) -> ChunkingResult:
    """Chunk a single filing's stored primary document body.

    Loads the filing + its FilingDocument, fetches the body bytes via the
    storage backend, chunks them, and replaces any prior chunks for this
    filing in one transaction. If the filing has no body row (the
    ``--with-bodies`` ingest hasn't run yet), returns a ``skipped_reason``
    instead of raising — batch jobs prefer skips over per-item exceptions.
    """

    pipeline = pipeline or ChunkingPipeline()

    filing_q = await session.execute(select(Filing).where(Filing.id == filing_id))
    filing = filing_q.scalar_one_or_none()
    if filing is None:
        return ChunkingResult(
            filing_id=filing_id,
            chunks_written=0,
            chunks_replaced=0,
            skipped_reason="filing_not_found",
        )

    document_q = await session.execute(
        select(FilingDocument).where(FilingDocument.filing_id == filing_id)
    )
    document = document_q.scalar_one_or_none()
    if document is None:
        return ChunkingResult(
            filing_id=filing_id,
            chunks_written=0,
            chunks_replaced=0,
            skipped_reason="no_body_stored",
        )

    body = await storage.get(document.storage_uri)
    chunks = pipeline.chunk_html(body)

    existing = await _load_chunks(session, filing_id)
    if existing:
        await session.execute(delete(FilingChunk).where(FilingChunk.filing_id == filing_id))

    rows = [
        FilingChunk(
            filing_id=filing_id,
            filing_date=filing.filing_date,
            ordinal=chunk.ordinal,
            section=chunk.section,
            text=chunk.text,
            token_count=chunk.token_count,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
        )
        for chunk in chunks
    ]
    session.add_all(rows)

    return ChunkingResult(
        filing_id=filing_id,
        chunks_written=len(rows),
        chunks_replaced=len(existing),
    )


async def chunk_filings_for_cik(
    storage: StorageBackend,
    cik: str,
    *,
    pipeline: ChunkingPipeline | None = None,
    config: ChunkingConfig | None = None,
    limit: int | None = None,
) -> list[ChunkingResult]:
    """Chunk every body-stored filing for ``cik``.

    Per-filing failures are caught and surfaced via ``skipped_reason`` /
    raised cleanly in logs; one bad filing doesn't abort the batch.
    """

    pipeline = pipeline or ChunkingPipeline(config=config)
    padded_cik = cik.strip().zfill(10)
    results: list[ChunkingResult] = []

    async with session_scope() as session:
        company_q = await session.execute(select(Company).where(Company.cik == padded_cik))
        company = company_q.scalar_one_or_none()
        if company is None:
            raise LookupError(f"company not ingested for cik={padded_cik!r}")

        filings_stmt = (
            select(Filing)
            .where(Filing.company_id == company.id)
            .order_by(Filing.filing_date.desc())
        )
        if limit is not None:
            filings_stmt = filings_stmt.limit(limit)

        filings = list((await session.execute(filings_stmt)).scalars())

        for filing in filings:
            try:
                result = await chunk_filing(
                    storage=storage,
                    session=session,
                    filing_id=filing.id,
                    pipeline=pipeline,
                )
            except Exception:
                logger.exception(
                    "chunking failed cik=%s accession=%s",
                    padded_cik,
                    filing.accession_number,
                )
                results.append(
                    ChunkingResult(
                        filing_id=filing.id,
                        chunks_written=0,
                        chunks_replaced=0,
                        skipped_reason="exception",
                    )
                )
                continue
            results.append(result)

    logger.info(
        "chunking cik=%s filings=%d total_chunks=%d",
        padded_cik,
        len(results),
        sum(r.chunks_written for r in results),
    )
    return results


__all__ = [
    "Chunk",
    "ChunkingConfig",
    "ChunkingPipeline",
    "ChunkingResult",
    "chunk_filing",
    "chunk_filings_for_cik",
]
