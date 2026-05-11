"""Reciprocal Rank Fusion (RRF) — score-free fusion of ranked lists.

RRF is the standard way to combine results from heterogeneous retrievers
where the raw scores are not directly comparable (BM25 and cosine
similarity, in our case). The fused score for a document ``d`` is:

    RRF(d) = sum over rankings r: 1 / (k + rank_r(d))

where ``rank_r(d)`` is the 1-indexed position of ``d`` in ranking ``r``.
Documents absent from a ranking contribute 0. ``k=60`` is the value from
Cormack et al. (2009) and is a reasonable default — the larger ``k`` is,
the less the top of a ranking dominates the fused order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

# Cormack et al. (2009) constant. Independent of the underlying retrievers;
# tune only if a downstream evaluation suite says you should.
DEFAULT_RRF_K = 60

T = TypeVar("T")


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[T]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[T, float]]:
    """Fuse ``rankings`` into a single ordered list under RRF scoring.

    Parameters
    ----------
    rankings:
        Each inner sequence is a ranking — most relevant item first. Items
        may be of any hashable type (typically chunk IDs).
    k:
        Smoothing constant. Must be non-negative. Larger values flatten the
        contribution of early ranks.

    Returns
    -------
    A list of ``(item, fused_score)`` pairs sorted by score descending. Ties
    are broken by the item's first appearance order across the input
    rankings, which is deterministic.
    """

    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    scores: dict[T, float] = {}
    first_seen: dict[T, int] = {}
    counter = 0

    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            if item not in first_seen:
                first_seen[item] = counter
                counter += 1

    return sorted(
        scores.items(),
        key=lambda kv: (-kv[1], first_seen[kv[0]]),
    )


__all__ = ["DEFAULT_RRF_K", "reciprocal_rank_fusion"]
