"""End-to-end: ``embed_chunks_for_filing`` writes pgvector data to the column.

Unit tests cover the service's control flow against a fake session. This
exercises the actual ``vector(384)`` column: that pgvector accepts the
list-of-floats payload SQLAlchemy hands it, that a cosine query against
the HNSW index returns the inserted rows, and that re-running the service
skips already-embedded chunks.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from alphamind.models.filing_chunk import EMBEDDING_DIM, FilingChunk
from alphamind.retrieval.embeddings.deterministic import DeterministicHashEmbedder
from alphamind.retrieval.embeddings.service import embed_chunks_for_filing
from tests.integration.conftest import (
    make_chunk,
    make_company,
    make_filing,
)

pytestmark = pytest.mark.integration


async def _seed_chunks(
    db_session: AsyncSession,
    *,
    texts: list[str],
) -> int:
    company = await make_company(db_session)
    filing = await make_filing(db_session, company=company)
    for i, body in enumerate(texts):
        await make_chunk(db_session, filing=filing, ordinal=i, text_body=body)
    await db_session.commit()
    return filing.id


async def test_embed_writes_vector_into_pgvector_column(db_session: AsyncSession) -> None:
    filing_id = await _seed_chunks(
        db_session,
        texts=["first chunk text", "second chunk text"],
    )

    result = await embed_chunks_for_filing(
        embedder=DeterministicHashEmbedder(),
        session=db_session,
        filing_id=filing_id,
    )
    await db_session.commit()

    assert result.chunks_embedded == 2
    assert result.chunks_skipped == 0

    rows = (
        (await db_session.execute(select(FilingChunk).where(FilingChunk.filing_id == filing_id)))
        .scalars()
        .all()
    )

    assert len(rows) == 2
    for row in rows:
        # pgvector round-trips as a list of floats matching the column dim.
        assert row.embedding is not None
        assert len(row.embedding) == EMBEDDING_DIM
        # Embedder produces unit-norm vectors; pgvector preserves that.
        norm = sum(v * v for v in row.embedding) ** 0.5
        assert 0.99 <= norm <= 1.01


async def test_embed_is_idempotent_on_replay(db_session: AsyncSession) -> None:
    filing_id = await _seed_chunks(
        db_session,
        texts=["the only chunk here"],
    )

    first = await embed_chunks_for_filing(
        embedder=DeterministicHashEmbedder(),
        session=db_session,
        filing_id=filing_id,
    )
    await db_session.commit()
    second = await embed_chunks_for_filing(
        embedder=DeterministicHashEmbedder(),
        session=db_session,
        filing_id=filing_id,
    )
    await db_session.commit()

    assert first.chunks_embedded == 1
    assert second.chunks_embedded == 0
    assert second.chunks_skipped == 1


async def test_force_re_embeds_even_when_present(db_session: AsyncSession) -> None:
    filing_id = await _seed_chunks(
        db_session,
        texts=["force me"],
    )

    await embed_chunks_for_filing(
        embedder=DeterministicHashEmbedder(),
        session=db_session,
        filing_id=filing_id,
    )
    await db_session.commit()

    forced = await embed_chunks_for_filing(
        embedder=DeterministicHashEmbedder(),
        session=db_session,
        filing_id=filing_id,
        force=True,
    )
    await db_session.commit()

    assert forced.chunks_embedded == 1
    assert forced.chunks_skipped == 0


async def test_hnsw_cosine_query_returns_embedded_rows(db_session: AsyncSession) -> None:
    """Smoke-test the HNSW index — a similarity query over written rows
    must return at least one match."""
    filing_id = await _seed_chunks(
        db_session,
        texts=["alpha beta gamma delta epsilon"],
    )

    embedder = DeterministicHashEmbedder()
    await embed_chunks_for_filing(
        embedder=embedder,
        session=db_session,
        filing_id=filing_id,
    )
    await db_session.commit()

    # Same string → same vector under the deterministic embedder. We
    # query for it and expect to get the chunk back at cosine distance 0.
    [query_vec] = await embedder.embed(["alpha beta gamma delta epsilon"])

    distance = FilingChunk.embedding.cosine_distance(query_vec).label("distance")
    rows = (
        await db_session.execute(
            select(FilingChunk.id, distance)
            .where(FilingChunk.embedding.is_not(None))
            .order_by(distance.asc())
            .limit(5)
        )
    ).all()

    assert len(rows) == 1
    # cosine_distance of identical unit vectors is 0 (allow a tiny float drift).
    assert rows[0][1] < 1e-6


async def test_embed_handles_empty_filing_gracefully(db_session: AsyncSession) -> None:
    company = await make_company(db_session)
    filing = await make_filing(db_session, company=company)
    await db_session.commit()

    result = await embed_chunks_for_filing(
        embedder=DeterministicHashEmbedder(),
        session=db_session,
        filing_id=filing.id,
    )

    assert result.chunks_embedded == 0
    assert result.chunks_skipped == 0
    # No chunks → no rows written, no errors raised.
    assert (
        await db_session.execute(
            select(func.count()).select_from(FilingChunk).where(FilingChunk.filing_id == filing.id)
        )
    ).scalar_one() == 0
