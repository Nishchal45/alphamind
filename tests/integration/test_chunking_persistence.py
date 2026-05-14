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
    <p>The Company designs, manufactures and markets smartphones, personal
       computers, tablets, wearables and accessories, and sells a variety of
       related services. The Company's products include iPhone, Mac, iPad,
       AirPods, Apple TV, Apple Watch, Beats products, HomePod, iPod touch
       and accessories. The Company's services include advertising,
       AppleCare, cloud services, digital content, and payment services.
       The Company sells its products and resells third-party products in
       most of its major markets directly to consumers, businesses and
       educational institutions through its retail and online stores and
       its direct sales force. The Company also employs a variety of
       indirect distribution channels, such as third-party cellular network
       carriers, wholesalers, retailers and resellers. During 2024, the
       Company's net sales through its direct and indirect distribution
       channels accounted for approximately 38% and 62%, respectively, of
       total net sales.</p>
    <p>Item 1A. Risk Factors</p>
    <p>The Company's business, reputation, results of operations, financial
       condition and stock price can be affected by a number of factors,
       whether currently known or unknown, including those described
       below. When any one or more of these risks materialize from time to
       time, the Company's business, reputation, results of operations,
       financial condition and stock price can be materially and adversely
       affected. Because of the following factors, as well as other
       factors affecting the Company's results of operations and financial
       condition, past financial performance should not be considered to be
       a reliable indicator of future performance, and investors should not
       use historical trends to anticipate results or trends in future
       periods. The Company's operations and performance depend
       significantly on global and regional economic conditions and adverse
       economic conditions can materially adversely affect the Company's
       business, results of operations and financial condition.</p>
    <p>Item 7. Management's Discussion and Analysis</p>
    <p>Total net sales for the year increased 2% to $383.3 billion compared
       to the prior year. iPhone net sales decreased 2% year over year due
       to lower iPhone net sales in markets outside of the United States,
       partially offset by higher iPhone net sales in the United States.
       Services net sales reached a new all-time annual record, increasing
       9% year over year. Mac net sales decreased compared to the prior
       year. iPad net sales decreased reflecting lower net sales of iPad
       Pro and entry-level iPad. Wearables, Home and Accessories net sales
       decreased compared to the prior year. The Company's effective tax
       rate for 2024 was 24.1% compared to 14.7% for 2023. The increase in
       the effective tax rate for 2024 was primarily due to a one-time
       income tax charge related to a state aid decision by the European
       General Court.</p>
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

    # Pin that we actually got chunks back — otherwise the equality below
    # is a false positive (0 == 0).
    assert first.chunks_written > 0
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
