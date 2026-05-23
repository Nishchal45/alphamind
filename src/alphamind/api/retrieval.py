"""Wire :class:`HybridSearch` into the agent layer's :data:`RetrievalFn`.

The agent graph factory takes a callable ``async def retrieve(*, query,
as_of, top_k) -> list[Source]``. Both ``scripts/research.py`` and the
API need to build one from the same components — :class:`HybridSearch`
for the retrieval itself and the ORM models for ticker / form metadata
the :class:`Source` dataclass requires.

This module exposes the wiring as a reusable factory so the API and
the CLI don't drift. ``scripts/research.py`` keeps its private copy
for now; consolidating it is a follow-up if the two implementations
diverge.
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
from alphamind.retrieval.search.pipeline import HybridSearch


async def _hydrate_metadata(
    session: AsyncSession,
    chunk_ids: list[int],
    *,
    as_of: date,
) -> dict[int, tuple[str, str]]:
    """Look up ``(ticker, form)`` for every chunk id in one query.

    ``ticker`` falls back to the CIK string when null so the citation
    header in :class:`Source` is never empty.
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


def build_hybrid_retrieval(search: HybridSearch) -> RetrievalFn:
    """Return a :data:`RetrievalFn` backed by the supplied ``HybridSearch``.

    Each call opens its own session because LangGraph fans specialists
    out in parallel — sharing a session across them would serialise the
    DB round-trips. The session lifetime is one retrieval call.
    """

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        async with session_scope() as session:
            hits = await search.search(
                session,
                query=query,
                as_of=as_of,
                top_k=top_k,
            )
            metadata = await _hydrate_metadata(
                session,
                [h.chunk_id for h in hits],
                as_of=as_of,
            )

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


__all__ = ["build_hybrid_retrieval"]
