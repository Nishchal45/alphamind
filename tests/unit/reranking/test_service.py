"""Unit tests for :func:`rerank_results`."""

from __future__ import annotations

from datetime import date

import pytest

from alphamind.reranking.base import RerankedPassage
from alphamind.reranking.service import rerank_results
from alphamind.retrieval.results import RetrievalResult

pytestmark = pytest.mark.asyncio


def _result(chunk_id: int, score: float, *, dense_rank: int | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        filing_document_id=10 + chunk_id,
        filing_id=100 + chunk_id,
        cik="0000000001",
        ticker="TST",
        company_name="Test Co",
        form="10-K",
        filing_date=date(2025, 1, 1),
        accession_number=f"0000000001-25-{chunk_id:06d}",
        section_label="preamble",
        section_title=None,
        chunk_index=chunk_id,
        text=f"text {chunk_id}",
        score=score,
        dense_rank=dense_rank,
    )


class _FakeReranker:
    model_name = "fake"

    def __init__(self, mapping: dict[int, float]) -> None:
        self._mapping = mapping

    async def rerank(
        self,
        query: str,
        passages: list[tuple[int, str]],
    ) -> list[RerankedPassage]:
        scored = [RerankedPassage(chunk_id=cid, score=self._mapping[cid]) for cid, _ in passages]
        scored.sort(key=lambda r: -r.score)
        return scored


async def test_replaces_score_and_resorts() -> None:
    inputs = [_result(1, 0.9), _result(2, 0.8), _result(3, 0.7)]
    reranker = _FakeReranker({1: 0.1, 2: 0.9, 3: 0.5})

    out = await rerank_results(reranker=reranker, query="q", results=inputs)

    assert [r.chunk_id for r in out] == [2, 3, 1]
    assert [r.score for r in out] == [0.9, 0.5, 0.1]


async def test_preserves_per_retriever_ranks() -> None:
    inputs = [_result(1, 0.9, dense_rank=1), _result(2, 0.8, dense_rank=2)]
    reranker = _FakeReranker({1: 0.2, 2: 0.8})

    out = await rerank_results(reranker=reranker, query="q", results=inputs)

    by_id = {r.chunk_id: r for r in out}
    assert by_id[1].dense_rank == 1
    assert by_id[2].dense_rank == 2


async def test_truncates_to_top_k() -> None:
    inputs = [_result(i, 0.0) for i in range(1, 6)]
    reranker = _FakeReranker({i: float(i) for i in range(1, 6)})

    out = await rerank_results(reranker=reranker, query="q", results=inputs, top_k=2)

    assert len(out) == 2
    assert [r.chunk_id for r in out] == [5, 4]


async def test_empty_input_returns_empty_without_calling_reranker() -> None:
    called = False

    class TripwireReranker:
        model_name = "tripwire"

        async def rerank(
            self,
            query: str,
            passages: list[tuple[int, str]],
        ) -> list[RerankedPassage]:
            nonlocal called
            called = True
            return []

    out = await rerank_results(reranker=TripwireReranker(), query="q", results=[])

    assert out == []
    assert called is False


async def test_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        await rerank_results(
            reranker=_FakeReranker({}),
            query="q",
            results=[_result(1, 0.5)],
            top_k=0,
        )


async def test_skips_chunks_the_reranker_invents() -> None:
    """If the reranker returns a chunk_id we did not pass in, skip it rather than crash."""

    class GhostReranker:
        model_name = "ghost"

        async def rerank(
            self,
            query: str,
            passages: list[tuple[int, str]],
        ) -> list[RerankedPassage]:
            return [
                RerankedPassage(chunk_id=999, score=10.0),  # not in input
                RerankedPassage(chunk_id=passages[0][0], score=1.0),
            ]

    inputs = [_result(7, 0.5)]
    out = await rerank_results(reranker=GhostReranker(), query="q", results=inputs)

    assert [r.chunk_id for r in out] == [7]
