"""End-to-end: HTML body in storage → ``chunk_filing`` → rows in filing_chunks.

Verifies the parts of the chunking persistence path that unit tests can't:

- Generated ``text_tsv`` column is materialised by Postgres.
- ``filing_date`` is denormalised onto each chunk (the time-horizon hot
  path; if this regresses, every retrieval query gets slower).
- Re-chunking is idempotent — the second call replaces the first chunk set
  without leaving orphans.
- The HNSW index on ``embedding`` doesn't reject inserts of NULL vectors.
"""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from alphamind.models.filing_chunk import FilingChunk
from alphamind.retrieval.chunking.service import chunk_filing
from alphamind.storage.local import LocalFilesystemStorage
from tests.integration.conftest import (
    make_company,
    make_filing,
    make_filing_document,
)

pytestmark = pytest.mark.integration

_SAMPLE_HTML = b"""
<html>
  <body>
    <p>Item 1. Business</p>
    <p>We design, manufacture and market smartphones, personal computers,
       tablets, wearables and accessories.</p>
    <p>Item 1A. Risk Factors</p>
    <p>The Company's business, reputation, results of operations, financial
       condition and stock price can be affected by a number of factors,
       whether currently known or unknown.</p>
    <p>Item 7. Management's Discussion and Analysis</p>
    <p>Total net sales for the year increased 2% to $383.3 billion. iPhone
       net sales decreased 2% year over year. Services net sales reached a
       new all-time annual record.</p>
  </body>
</html>
"""


async def _seed_filing_with_body(
    *,
    session: AsyncSession,
    storage: LocalFilesystemStorage,
    body: bytes = _SAMPLE_HTML,
) -> int:
    company = await make_company(session)
    filing = await make_filing(session, company=company)
    digest = hashlib.sha256(body).hexdigest()
    uri = await storage.put(key=digest, data=body)
    await make_filing_document(
        session,
        filing=filing,
        storage_uri=uri,
        content_hash=digest,
        byte_size=len(body),
    )
    await session.commit()
    return filing.id


async def test_chunk_filing_writes_rows_with_generated_tsvector(
    db_session: AsyncSession,
    storage: LocalFilesystemStorage,
) -> None:
    filing_id = await _seed_filing_with_body(session=db_session, storage=storage)

    result = await chunk_filing(
        storage=storage,
        session=db_session,
        filing_id=filing_id,
    )
    await db_session.commit()

    assert result.chunks_written > 0
    assert result.chunks_replaced == 0

    rows = (
        (
            await db_session.execute(
                select(FilingChunk)
                .where(FilingChunk.filing_id == filing_id)
                .order_by(FilingChunk.ordinal)
            )
        )
        .scalars()
        .all()
    )

    assert len(rows) == result.chunks_written
    assert all(row.token_count > 0 for row in rows)
    # Sections survive the round trip (at least one chunk picks up a heading).
    assert any(row.section is not None for row in rows)

    # The text_tsv generated column is materialised — a tsquery match
    # against a known term should hit at least one chunk.
    hit_count = (
        await db_session.execute(
            select(func.count())
            .select_from(FilingChunk)
            .where(
                FilingChunk.filing_id == filing_id,
                FilingChunk.text_tsv.op("@@")(func.plainto_tsquery("english", "smartphones")),
            )
        )
    ).scalar_one()
    assert hit_count >= 1


async def test_chunk_filing_denormalises_filing_date_onto_each_chunk(
    db_session: AsyncSession,
    storage: LocalFilesystemStorage,
) -> None:
    filing_id = await _seed_filing_with_body(session=db_session, storage=storage)

    await chunk_filing(storage=storage, session=db_session, filing_id=filing_id)
    await db_session.commit()

    rows = (
        (await db_session.execute(select(FilingChunk).where(FilingChunk.filing_id == filing_id)))
        .scalars()
        .all()
    )

    # Every chunk carries the parent filing's date (the time-horizon
    # predicate has to be cheap and index-friendly per ADR 0005).
    parent_dates = {row.filing_date for row in rows}
    assert len(parent_dates) == 1
    assert parent_dates == {rows[0].filing_date}


async def test_chunk_filing_is_idempotent_on_replay(
    db_session: AsyncSession,
    storage: LocalFilesystemStorage,
) -> None:
    filing_id = await _seed_filing_with_body(session=db_session, storage=storage)

    first = await chunk_filing(storage=storage, session=db_session, filing_id=filing_id)
    await db_session.commit()

    second = await chunk_filing(storage=storage, session=db_session, filing_id=filing_id)
    await db_session.commit()

    # Same body → same chunks count. Second pass replaces the first.
    assert second.chunks_written == first.chunks_written
    assert second.chunks_replaced == first.chunks_written

    total = (
        await db_session.execute(
            select(func.count()).select_from(FilingChunk).where(FilingChunk.filing_id == filing_id)
        )
    ).scalar_one()
    # No orphans: total rows = chunks_written, not 2x that.
    assert total == first.chunks_written


async def test_chunk_filing_skips_when_no_body_is_stored(
    db_session: AsyncSession,
    storage: LocalFilesystemStorage,
) -> None:
    """A filing with no FilingDocument should skip gracefully, not raise."""
    company = await make_company(db_session)
    filing = await make_filing(db_session, company=company)
    await db_session.commit()

    result = await chunk_filing(
        storage=storage,
        session=db_session,
        filing_id=filing.id,
    )

    assert result.chunks_written == 0
    assert result.skipped_reason == "no_body_stored"
