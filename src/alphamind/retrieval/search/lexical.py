"""BM25 / lexical search over ``filing_chunks.text_tsv``.

Uses Postgres' ``ts_rank_cd`` against a ``plainto_tsquery``-compiled query.
Cheap, sturdy, and good at exact-term retrieval (ticker symbols, company
names, regulatory phrases that don't paraphrase well). Pairs nicely with
dense retrieval, which is good at the cases lexical isn't.

The time-horizon predicate (``filing_date <= :as_of``) is applied here in
the WHERE clause so it short-circuits the GIN index scan rather than
filtering candidates after retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True, slots=True)
class LexicalHit:
    """One BM25 candidate. Score is ``ts_rank_cd``; higher is better."""

    chunk_id: int
    score: float


_LEXICAL_SQL = text(
    """
    SELECT id,
           ts_rank_cd(text_tsv, plainto_tsquery('english', :query)) AS score
    FROM filing_chunks
    WHERE text_tsv @@ plainto_tsquery('english', :query)
      AND filing_date <= :as_of
    ORDER BY score DESC
    LIMIT :limit
    """
)


async def lexical_search(
    session: AsyncSession,
    *,
    query: str,
    as_of: date,
    limit: int = 50,
) -> list[LexicalHit]:
    """Return up to ``limit`` chunks that match ``query`` lexically.

    The ``as_of`` filter is non-negotiable: if a chunk's filing post-dates
    the analysis horizon, it cannot appear in the results. This prevents
    lookahead bias during backtests and historical replays.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")
    if not query.strip():
        return []

    result = await session.execute(
        _LEXICAL_SQL,
        {"query": query, "as_of": as_of, "limit": limit},
    )
    return [LexicalHit(chunk_id=row[0], score=float(row[1])) for row in result.all()]
