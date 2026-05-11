"""Glue between :class:`Reranker` and :class:`RetrievalResult`.

The retrieval layer returns rich :class:`RetrievalResult` objects keyed on
``chunk_id``. The reranker protocol speaks ``(chunk_id, text)`` tuples and
returns ``RerankedPassage`` objects. :func:`rerank_results` is the small
adaptor that pipes one into the other: score each result against the
query, sort by reranker score, and return a new list with ``.score``
replaced.
"""

from __future__ import annotations

from dataclasses import replace

from alphamind.reranking.base import Reranker
from alphamind.retrieval.results import RetrievalResult


async def rerank_results(
    *,
    reranker: Reranker,
    query: str,
    results: list[RetrievalResult],
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Rerank ``results`` against ``query`` and optionally truncate to ``top_k``.

    ``score`` on the returned :class:`RetrievalResult` is overwritten with
    the reranker's score; the per-retriever ranks (``dense_rank``,
    ``bm25_rank``) are preserved so callers can still see which retriever
    surfaced each chunk in the candidate pool.
    """

    if not results:
        return []
    if top_k is not None and top_k <= 0:
        raise ValueError(f"top_k must be positive when set, got {top_k}")

    passages = [(r.chunk_id, r.text) for r in results]
    reranked = await reranker.rerank(query, passages)

    by_id = {r.chunk_id: r for r in results}
    out: list[RetrievalResult] = []
    for entry in reranked:
        base = by_id.get(entry.chunk_id)
        if base is None:
            # Shouldn't happen — the reranker returns the same chunk_ids
            # it was given — but skipping is safer than crashing in prod.
            continue
        out.append(replace(base, score=entry.score))
    if top_k is not None:
        out = out[:top_k]
    return out


__all__ = ["rerank_results"]
