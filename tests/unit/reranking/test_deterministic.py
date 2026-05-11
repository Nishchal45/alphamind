"""Unit tests for :class:`DeterministicReranker`.

These pin the contract the rest of the system relies on:

- Scoring is bounded (Jaccard ∈ [0, 1]) and deterministic across calls.
- Higher token overlap with the query yields a higher score.
- The returned list preserves every input chunk_id exactly once.
- Tie-breaks fall back to input order.
"""

from __future__ import annotations

import pytest

from alphamind.reranking.deterministic import DeterministicReranker

pytestmark = pytest.mark.asyncio


async def test_perfect_overlap_scores_one_no_overlap_scores_zero() -> None:
    reranker = DeterministicReranker()
    out = await reranker.rerank(
        "alpha beta gamma",
        [
            (1, "alpha beta gamma"),
            (2, "nothing in common here"),
        ],
    )

    by_id = {r.chunk_id: r.score for r in out}
    assert by_id[1] == pytest.approx(1.0)
    assert by_id[2] == pytest.approx(0.0)


async def test_more_overlap_outranks_less_overlap() -> None:
    reranker = DeterministicReranker()
    out = await reranker.rerank(
        "supply chain risk semiconductor exposure",
        [
            (10, "semiconductor exposure was a key supply risk"),  # heavy overlap
            (20, "we hosted an annual employee picnic"),  # essentially none
            (30, "supply chain disruptions affected shipping"),  # partial overlap
        ],
    )

    ordered_ids = [r.chunk_id for r in out]
    assert ordered_ids[0] == 10
    assert ordered_ids[-1] == 20


async def test_returns_all_input_chunks_exactly_once() -> None:
    reranker = DeterministicReranker()
    inputs: list[tuple[int, str]] = [(i, f"chunk {i} text") for i in range(5)]

    out = await reranker.rerank("chunk text", inputs)

    assert {r.chunk_id for r in out} == {i for i, _ in inputs}
    assert len(out) == len(inputs)


async def test_tie_breaks_by_input_order() -> None:
    reranker = DeterministicReranker()
    # Two passages with the same token set → tied score → input order wins.
    out = await reranker.rerank(
        "alpha beta",
        [
            (100, "alpha beta"),
            (200, "beta alpha"),
        ],
    )

    assert [r.chunk_id for r in out] == [100, 200]


async def test_empty_query_yields_zero_scores() -> None:
    reranker = DeterministicReranker()
    out = await reranker.rerank("", [(1, "some text"), (2, "other text")])

    assert all(r.score == 0.0 for r in out)


async def test_empty_passage_text_yields_zero_score() -> None:
    reranker = DeterministicReranker()
    out = await reranker.rerank("relevant query", [(1, "")])

    assert out[0].chunk_id == 1
    assert out[0].score == 0.0


async def test_deterministic_across_repeated_calls() -> None:
    reranker = DeterministicReranker()
    passages: list[tuple[int, str]] = [
        (1, "alpha beta"),
        (2, "beta gamma"),
        (3, "gamma delta"),
    ]

    first = await reranker.rerank("alpha", passages)
    second = await reranker.rerank("alpha", passages)

    assert [(r.chunk_id, r.score) for r in first] == [(r.chunk_id, r.score) for r in second]


async def test_model_name_is_stable() -> None:
    assert DeterministicReranker().model_name == "deterministic-jaccard"
