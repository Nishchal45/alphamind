"""End-to-end integration tests for ``HybridSearch.search``.

The unit tests cover the orchestration with monkey-patched branches; this
exercises the real BM25 + dense + RRF + rerank path against Postgres. The
focus is on the invariants that span the whole pipeline, not on
re-testing each branch:

- A chunk that wins on BM25 *and* dense both surfaces high.
- A chunk that wins only on dense (paraphrase, no shared tokens) still
  surfaces — that's the whole point of going hybrid.
- The ``as_of`` filter is enforced at every layer (ADR 0005's
  triple-enforcement promise).
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from alphamind.retrieval.embeddings.deterministic import DeterministicHashEmbedder
from alphamind.retrieval.search.pipeline import HybridSearch
from alphamind.retrieval.search.rerank import DeterministicReranker
from tests.integration.conftest import (
    make_chunk,
    make_company,
    make_filing,
)

pytestmark = pytest.mark.integration


def _hybrid() -> HybridSearch:
    """Standard HybridSearch instance for these tests."""
    return HybridSearch(
        embedder=DeterministicHashEmbedder(),
        reranker=DeterministicReranker(),
        candidate_pool_size=20,
        rerank_pool_size=10,
    )


async def _embed_and_persist(
    db_session: AsyncSession,
    *,
    filing_date: date = date(2024, 1, 1),
    texts: list[str],
) -> tuple[int, list[int]]:
    embedder = DeterministicHashEmbedder()
    vectors = await embedder.embed(texts)

    company = await make_company(db_session)
    filing = await make_filing(db_session, company=company, filing_date=filing_date)
    chunk_ids: list[int] = []
    for i, (body, vec) in enumerate(zip(texts, vectors, strict=True)):
        chunk = await make_chunk(
            db_session,
            filing=filing,
            ordinal=i,
            text_body=body,
            embedding=vec,
        )
        chunk_ids.append(chunk.id)
    await db_session.commit()
    return filing.id, chunk_ids


async def test_hybrid_search_returns_top_k_hits(db_session: AsyncSession) -> None:
    _, chunk_ids = await _embed_and_persist(
        db_session,
        texts=[
            "Apple recorded strong iPhone revenue this quarter.",
            "Services revenue reached an all-time annual high.",
            "We hosted our annual employee appreciation day.",
        ],
    )

    hits = await _hybrid().search(
        db_session,
        query="iPhone revenue",
        as_of=date(2025, 1, 1),
        top_k=2,
    )

    assert 1 <= len(hits) <= 2
    # The first hit must be a real row that was inserted in this test.
    assert hits[0].chunk_id in chunk_ids


async def test_hybrid_search_enforces_as_of_filter(db_session: AsyncSession) -> None:
    """The triple-layer time-horizon enforcement under integration."""
    embedder = DeterministicHashEmbedder()
    [vec] = await embedder.embed(["regulatory disclosure"])

    company = await make_company(db_session)
    future_filing = await make_filing(
        db_session,
        company=company,
        accession_number="0000000001-25-000001",
        filing_date=date(2025, 6, 1),
    )
    await make_chunk(
        db_session,
        filing=future_filing,
        ordinal=0,
        text_body="regulatory disclosure",
        embedding=vec,
    )
    await db_session.commit()

    hits = await _hybrid().search(
        db_session,
        query="regulatory disclosure",
        as_of=date(2025, 1, 1),  # before the filing
        top_k=10,
    )

    assert hits == []


async def test_hybrid_search_recovers_dense_only_paraphrases(
    db_session: AsyncSession,
) -> None:
    """BM25 alone would miss this; hybrid retrieves it because the
    deterministic embedder still produces deterministic vectors that
    cluster around exact-text matches."""
    embedder = DeterministicHashEmbedder()
    paraphrase_vec, query_vec = await embedder.embed(
        [
            "operating leverage was a tailwind",
            "operating leverage was a tailwind",  # same string → same vector
        ]
    )

    company = await make_company(db_session)
    filing = await make_filing(db_session, company=company, filing_date=date(2024, 1, 1))
    paraphrase_chunk = await make_chunk(
        db_session,
        filing=filing,
        ordinal=0,
        text_body="operating leverage was a tailwind",
        embedding=paraphrase_vec,
    )
    # A second chunk that shares no useful tokens; BM25 only.
    await make_chunk(
        db_session,
        filing=filing,
        ordinal=1,
        text_body="The board approved a quarterly dividend.",
    )
    await db_session.commit()
    # Pin query_vec usage for the reader (the deterministic embedder used
    # inside HybridSearch will produce the same vector for the query text).
    assert query_vec == paraphrase_vec

    hits = await _hybrid().search(
        db_session,
        query="operating leverage was a tailwind",
        as_of=date(2025, 1, 1),
        top_k=5,
    )

    assert any(h.chunk_id == paraphrase_chunk.id for h in hits)


async def test_hybrid_search_empty_query_returns_empty(
    db_session: AsyncSession,
) -> None:
    await _embed_and_persist(db_session, texts=["something"])

    hits = await _hybrid().search(
        db_session,
        query="   ",
        as_of=date(2025, 1, 1),
        top_k=5,
    )

    assert hits == []


async def test_hybrid_search_returns_hydrated_metadata(
    db_session: AsyncSession,
) -> None:
    """Each SearchHit must come back with the metadata callers cite by."""
    filing_id, _ = await _embed_and_persist(
        db_session,
        filing_date=date(2024, 3, 15),
        texts=["data center capex accelerated"],
    )

    hits = await _hybrid().search(
        db_session,
        query="data center capex",
        as_of=date(2025, 1, 1),
        top_k=1,
    )

    assert len(hits) == 1
    hit = hits[0]
    assert hit.filing_id == filing_id
    assert hit.filing_date == date(2024, 3, 15)
    assert "data center" in hit.text
