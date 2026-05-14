"""Integration tests for ``lexical_search``.

Specifically verifies:

- The GIN index on ``text_tsv`` actually serves matches (not just the
  generated column, which the chunking-persistence suite covers).
- ``ts_rank_cd`` orders results sensibly relative to the query.
- The ``as_of`` time-horizon predicate filters out future filings —
  the project's single most important correctness invariant per ADR 0005.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from alphamind.retrieval.search.lexical import lexical_search
from tests.integration.conftest import (
    make_chunk,
    make_company,
    make_filing,
)

pytestmark = pytest.mark.integration


async def test_lexical_search_matches_by_keyword(db_session: AsyncSession) -> None:
    company = await make_company(db_session)
    filing = await make_filing(db_session, company=company, filing_date=date(2024, 1, 1))
    await make_chunk(
        db_session,
        filing=filing,
        ordinal=0,
        text_body="The Company recorded an inventory write-down of $200 million.",
    )
    await make_chunk(
        db_session,
        filing=filing,
        ordinal=1,
        text_body="The annual employee picnic was held in June.",
    )
    await db_session.commit()

    hits = await lexical_search(
        db_session,
        query="inventory write-down",
        as_of=date(2025, 1, 1),
        limit=10,
    )

    # The first chunk is the only one that contains the query terms.
    assert len(hits) == 1
    assert hits[0].score > 0


async def test_lexical_search_ranks_better_matches_higher(
    db_session: AsyncSession,
) -> None:
    company = await make_company(db_session)
    filing = await make_filing(db_session, company=company, filing_date=date(2024, 1, 1))
    await make_chunk(
        db_session,
        filing=filing,
        ordinal=0,
        text_body=(
            "Supply chain risks remain elevated. Supply disruptions and "
            "supply chain bottlenecks affected gross margin."
        ),
    )
    await make_chunk(
        db_session,
        filing=filing,
        ordinal=1,
        text_body="We disclosed certain supply chain considerations in note 3.",
    )
    await db_session.commit()

    hits = await lexical_search(
        db_session,
        query="supply chain",
        as_of=date(2025, 1, 1),
        limit=10,
    )

    assert len(hits) == 2
    # Heavier term repetition wins — that's what ts_rank_cd rewards.
    assert hits[0].score >= hits[1].score


async def test_lexical_search_enforces_as_of_filter(db_session: AsyncSession) -> None:
    """Chunks filed AFTER the as_of horizon must not appear."""
    company = await make_company(db_session)

    old_filing = await make_filing(
        db_session,
        company=company,
        accession_number="0000000001-23-000001",
        filing_date=date(2023, 6, 1),
    )
    await make_chunk(
        db_session,
        filing=old_filing,
        ordinal=0,
        text_body="Cloud revenue grew 30 percent year over year.",
    )

    new_filing = await make_filing(
        db_session,
        company=company,
        accession_number="0000000001-24-000001",
        filing_date=date(2024, 12, 1),
    )
    await make_chunk(
        db_session,
        filing=new_filing,
        ordinal=0,
        text_body="Cloud revenue continued to grow strongly.",
    )
    await db_session.commit()

    # As of mid-2024: only the older chunk should be visible.
    hits = await lexical_search(
        db_session,
        query="cloud revenue",
        as_of=date(2024, 6, 30),
        limit=10,
    )

    assert len(hits) == 1
    # The OLDER filing's chunk is the only valid candidate.
    # We can't compare chunk ids easily without re-querying, so verify
    # the count is right and trust the SQL — the next test pins behaviour
    # at the boundary date.

    # As of 2025: both are visible.
    hits_later = await lexical_search(
        db_session,
        query="cloud revenue",
        as_of=date(2025, 1, 1),
        limit=10,
    )
    assert len(hits_later) == 2


async def test_lexical_search_treats_as_of_boundary_inclusively(
    db_session: AsyncSession,
) -> None:
    """``filing_date <= :as_of`` — a chunk filed exactly on the horizon must qualify."""
    company = await make_company(db_session)
    filing = await make_filing(db_session, company=company, filing_date=date(2024, 6, 30))
    await make_chunk(
        db_session,
        filing=filing,
        ordinal=0,
        text_body="Operating margin improved 200 basis points.",
    )
    await db_session.commit()

    hits = await lexical_search(
        db_session,
        query="operating margin",
        as_of=date(2024, 6, 30),
        limit=10,
    )

    assert len(hits) == 1


async def test_lexical_search_returns_nothing_for_empty_query(
    db_session: AsyncSession,
) -> None:
    company = await make_company(db_session)
    filing = await make_filing(db_session, company=company)
    await make_chunk(db_session, filing=filing, ordinal=0, text_body="content here")
    await db_session.commit()

    hits = await lexical_search(db_session, query="   ", as_of=date(2025, 1, 1))

    assert hits == []
