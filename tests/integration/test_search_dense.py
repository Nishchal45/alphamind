"""Integration tests for ``dense_search``.

Verifies the actual pgvector cosine query and the HNSW index path:

- Exact-match query vector finds the matching chunk first.
- Distinct chunks return distinct rows (HNSW index isn't deduplicating).
- ``as_of`` is enforced inside the same SELECT (pushed under the index scan).
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from alphamind.retrieval.embeddings.deterministic import DeterministicHashEmbedder
from alphamind.retrieval.search.dense import dense_search
from tests.integration.conftest import (
    make_chunk,
    make_company,
    make_filing,
)

pytestmark = pytest.mark.integration


async def _seed_two_chunks_with_embeddings(
    db_session: AsyncSession,
    *,
    filing_date: date = date(2024, 1, 1),
) -> tuple[int, int]:
    embedder = DeterministicHashEmbedder()
    vec_a, vec_b = await embedder.embed(["alpha alpha alpha", "beta beta beta"])

    company = await make_company(db_session)
    filing = await make_filing(db_session, company=company, filing_date=filing_date)
    a = await make_chunk(
        db_session,
        filing=filing,
        ordinal=0,
        text_body="alpha alpha alpha",
        embedding=vec_a,
    )
    b = await make_chunk(
        db_session,
        filing=filing,
        ordinal=1,
        text_body="beta beta beta",
        embedding=vec_b,
    )
    await db_session.commit()
    return a.id, b.id


async def test_dense_search_finds_nearest_chunk_first(db_session: AsyncSession) -> None:
    a_id, b_id = await _seed_two_chunks_with_embeddings(db_session)

    embedder = DeterministicHashEmbedder()
    [query_vec] = await embedder.embed(["alpha alpha alpha"])

    hits = await dense_search(
        db_session,
        query_vector=query_vec,
        as_of=date(2025, 1, 1),
        limit=5,
    )

    assert len(hits) == 2
    # The chunk whose text matches the query vector exactly should be #1.
    assert hits[0].chunk_id == a_id
    assert hits[1].chunk_id == b_id
    # And its similarity (1 - cosine_distance) should be effectively 1.
    assert hits[0].score > 0.999


async def test_dense_search_enforces_as_of_filter(db_session: AsyncSession) -> None:
    embedder = DeterministicHashEmbedder()
    [vec] = await embedder.embed(["future material"])

    company = await make_company(db_session)
    future_filing = await make_filing(
        db_session,
        company=company,
        accession_number="0000000001-25-000001",
        filing_date=date(2025, 1, 15),
    )
    await make_chunk(
        db_session,
        filing=future_filing,
        ordinal=0,
        text_body="future material",
        embedding=vec,
    )
    await db_session.commit()

    hits = await dense_search(
        db_session,
        query_vector=vec,
        as_of=date(2024, 12, 31),  # before the filing date
        limit=5,
    )

    assert hits == []


async def test_dense_search_skips_chunks_with_null_embedding(
    db_session: AsyncSession,
) -> None:
    """A chunk without an embedding can't be ranked and must not surface."""
    embedder = DeterministicHashEmbedder()
    [vec] = await embedder.embed(["only one"])

    company = await make_company(db_session)
    filing = await make_filing(db_session, company=company)
    await make_chunk(
        db_session,
        filing=filing,
        ordinal=0,
        text_body="this one is embedded",
        embedding=vec,
    )
    await make_chunk(
        db_session,
        filing=filing,
        ordinal=1,
        text_body="this one is NOT embedded yet",
        embedding=None,
    )
    await db_session.commit()

    hits = await dense_search(
        db_session,
        query_vector=vec,
        as_of=date(2025, 1, 1),
        limit=5,
    )

    assert len(hits) == 1


async def test_dense_search_respects_limit(db_session: AsyncSession) -> None:
    embedder = DeterministicHashEmbedder()
    vecs = await embedder.embed([f"chunk text {i}" for i in range(5)])

    company = await make_company(db_session)
    filing = await make_filing(db_session, company=company)
    for i, vec in enumerate(vecs):
        await make_chunk(
            db_session,
            filing=filing,
            ordinal=i,
            text_body=f"chunk text {i}",
            embedding=vec,
        )
    await db_session.commit()

    [query_vec] = await embedder.embed(["unrelated"])
    hits = await dense_search(
        db_session,
        query_vector=query_vec,
        as_of=date(2025, 1, 1),
        limit=3,
    )

    assert len(hits) == 3
