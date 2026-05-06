"""Reciprocal Rank Fusion across multiple ranked candidate lists.

Why RRF: it's the simplest fusion that works well in practice and doesn't
require score calibration between branches. ``ts_rank_cd`` and cosine
similarity live on completely different scales — comparing them directly
would let one branch dominate based on score magnitude rather than
relevance. RRF only looks at *rank*, which is calibration-free.

Reference: Cormack, Clarke, Buettcher (2009),
"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning
Methods."
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FusedHit:
    """One chunk's fused-rank score across all input branches."""

    chunk_id: int
    score: float


def reciprocal_rank_fusion(
    branches: Iterable[Sequence[Any]],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[FusedHit]:
    """Combine ranked hit lists using RRF.

    Parameters
    ----------
    branches:
        Iterable of ranked lists. Each list is treated as descending by
        relevance — the first hit is rank 1, the second rank 2, and so on.
    k:
        Smoothing constant. The original paper recommends 60. Larger
        values dampen the contribution of top-ranked items in any single
        branch; smaller values amplify it.
    limit:
        If given, return only the top ``limit`` fused hits.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")

    scores: dict[int, float] = {}
    for branch in branches:
        for rank, hit in enumerate(branch, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)

    fused = [FusedHit(chunk_id=cid, score=score) for cid, score in scores.items()]
    fused.sort(key=lambda h: h.score, reverse=True)

    if limit is not None:
        fused = fused[:limit]
    return fused
