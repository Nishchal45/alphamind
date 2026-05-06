"""Tests for the deterministic reranker."""

from __future__ import annotations

import pytest

from alphamind.retrieval.search.rerank import (
    DeterministicReranker,
    RerankCandidate,
    Reranker,
)

pytestmark = pytest.mark.asyncio


async def test_satisfies_reranker_protocol() -> None:
    assert isinstance(DeterministicReranker(), Reranker)


async def test_high_token_overlap_ranks_first() -> None:
    reranker = DeterministicReranker()
    candidates = [
        RerankCandidate(chunk_id=1, text="Apple inventory write-down details follow."),
        RerankCandidate(chunk_id=2, text="Sales of services rose 8% year over year."),
        RerankCandidate(chunk_id=3, text="Inventory write-down recorded for obsolete stock."),
    ]

    out = await reranker.rerank("inventory write-down", candidates)

    # Both 1 and 3 should rank ahead of 2; the one with more overlap first.
    ranked_ids = [h.chunk_id for h in out]
    assert ranked_ids[-1] == 2
    assert set(ranked_ids[:2]) == {1, 3}


async def test_empty_candidate_list_returns_empty() -> None:
    reranker = DeterministicReranker()
    assert await reranker.rerank("anything", []) == []


async def test_stable_for_ties() -> None:
    """Identical scores fall back to original input order."""
    reranker = DeterministicReranker()
    candidates = [
        RerankCandidate(chunk_id=10, text="alpha beta gamma"),
        RerankCandidate(chunk_id=20, text="alpha beta gamma"),
        RerankCandidate(chunk_id=30, text="alpha beta gamma"),
    ]

    out = await reranker.rerank("alpha beta", candidates)

    assert [h.chunk_id for h in out] == [10, 20, 30]


async def test_returns_one_score_per_candidate() -> None:
    reranker = DeterministicReranker()
    candidates = [RerankCandidate(chunk_id=i, text=f"sentence {i} about apples") for i in range(5)]

    out = await reranker.rerank("apples", candidates)

    assert len(out) == len(candidates)
    assert {h.chunk_id for h in out} == {c.chunk_id for c in candidates}
