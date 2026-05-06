"""Dense ANN search over ``filing_chunks.embedding`` via pgvector.

Cosine distance against the HNSW index built in migration 0004. Returns
the top-``limit`` chunks ordered by similarity (1 - distance), with the
time-horizon predicate applied as a hard WHERE clause so the planner can
push it under the ANN scan.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True, slots=True)
class DenseHit:
    """One ANN candidate. Score is cosine similarity in ``[-1, 1]``; higher is better."""

    chunk_id: int
    score: float


_DENSE_SQL = text(
    """
    SELECT id,
           1 - (embedding <=> CAST(:query_vec AS vector)) AS score
    FROM filing_chunks
    WHERE embedding IS NOT NULL
      AND filing_date <= :as_of
    ORDER BY embedding <=> CAST(:query_vec AS vector)
    LIMIT :limit
    """
)


def _format_vector(vec: Sequence[float]) -> str:
    """pgvector accepts a literal of the form '[1.0,2.0,3.0]'."""
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"


async def dense_search(
    session: AsyncSession,
    *,
    query_vector: Sequence[float],
    as_of: date,
    limit: int = 50,
) -> list[DenseHit]:
    """Return up to ``limit`` chunks ranked by cosine similarity to ``query_vector``."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    if not query_vector:
        raise ValueError("query_vector must be non-empty")

    result = await session.execute(
        _DENSE_SQL,
        {
            "query_vec": _format_vector(query_vector),
            "as_of": as_of,
            "limit": limit,
        },
    )
    return [DenseHit(chunk_id=row[0], score=float(row[1])) for row in result.all()]
