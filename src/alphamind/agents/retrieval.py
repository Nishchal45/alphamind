"""Adapter from :class:`HybridSearch` to the agent :data:`RetrievalFn` contract.

The agent graph is parameterised on a ``RetrievalFn`` so tests can
inject scripted retrieval. Production needs to wire that callable to
the SQLAlchemy + ``HybridSearch`` path. This module is the single
place that wiring lives — both ``scripts/research.py`` and the API
serving layer import :func:`build_filing_retrieval_fn` from here.

The function opens its own session per call. The DAG runs nodes
sequentially today so this keeps the lifetime simple; when concurrent
fan-out becomes interesting, callers can share a session by closing
over one in this module instead.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from alphamind.agents.specialists._base import RetrievalFn
from alphamind.agents.state import Source
from alphamind.db.session import session_scope
from alphamind.models.filing import Filing
from alphamind.models.filing_chunk import FilingChunk
from alphamind.retrieval.search import HybridSearch


async def _hydrate(
    session: AsyncSession,
    chunk_ids: list[int],
    *,
    as_of: date,
) -> dict[int, tuple[str, str]]:
    """Look up ``(ticker, form)`` for every chunk id.

    Returns a dict keyed by chunk_id so callers can stitch metadata
    back onto each :class:`Source` without round-tripping per chunk.
    ``ticker`` falls back to the CIK string when null so the citation
    header is never empty.
    """
    if not chunk_ids:
        return {}
    stmt = (
        select(FilingChunk)
        .where(FilingChunk.id.in_(chunk_ids))
        .where(FilingChunk.filing_date <= as_of)
        .options(selectinload(FilingChunk.filing).selectinload(Filing.company))
    )
    rows = (await session.execute(stmt)).scalars().all()
    out: dict[int, tuple[str, str]] = {}
    for row in rows:
        ticker = row.filing.company.ticker or row.filing.company.cik
        out[row.id] = (ticker, row.filing.form)
    return out


def build_filing_retrieval_fn(search: HybridSearch) -> RetrievalFn:
    """Return a :data:`RetrievalFn` backed by ``search`` and the filings DB."""

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        async with session_scope() as session:
            hits = await search.search(
                session,
                query=query,
                as_of=as_of,
                top_k=top_k,
            )
            metadata = await _hydrate(session, [h.chunk_id for h in hits], as_of=as_of)

        sources: list[Source] = []
        for hit in hits:
            ticker, form = metadata.get(hit.chunk_id, (str(hit.filing_id), "—"))
            sources.append(
                Source(
                    chunk_id=hit.chunk_id,
                    filing_id=hit.filing_id,
                    ticker=ticker,
                    form=form,
                    filing_date=hit.filing_date,
                    section=hit.section,
                    text=hit.text,
                    score=hit.score,
                )
            )
        return sources

    return retrieve


__all__ = ["build_filing_retrieval_fn"]
