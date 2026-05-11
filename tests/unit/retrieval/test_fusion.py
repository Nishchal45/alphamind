"""Unit tests for :func:`reciprocal_rank_fusion`.

These pin the math contract the rest of retrieval relies on: items present
in multiple rankings beat items present in only one; ties break by first-
appearance order across the input rankings; absent items contribute 0.
"""

from __future__ import annotations

import math

import pytest

from alphamind.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion


def test_single_ranking_preserves_order() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"]])

    assert [item for item, _ in fused] == ["a", "b", "c"]


def test_two_rankings_boost_overlapping_items() -> None:
    # 'b' appears at rank 2 in both → highest fused score.
    # 'a' appears only in the first ranking; 'c' only in the second.
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]])
    order = [item for item, _ in fused]

    assert order[0] == "b"
    # 'a' and 'c' tie on score (both at rank 1 in one ranking); first-seen
    # order across inputs breaks the tie deterministically.
    assert order[1] == "a"
    assert order[2] == "c"


def test_known_rrf_scores_with_default_k() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"]]))

    assert math.isclose(fused["a"], 1.0 / (DEFAULT_RRF_K + 1))
    assert math.isclose(fused["b"], 1.0 / (DEFAULT_RRF_K + 2))


def test_item_present_in_two_rankings_outranks_first_in_one() -> None:
    # 'a' is rank-1 in ranking A but absent from B.
    # 'b' is rank-2 in A and rank-1 in B.
    # Fused: a = 1/(60+1); b = 1/(60+2) + 1/(60+1). b wins.
    fused = reciprocal_rank_fusion([["a", "b"], ["b"]])

    order = [item for item, _ in fused]
    assert order == ["b", "a"]


def test_returns_empty_when_no_rankings() -> None:
    assert reciprocal_rank_fusion([]) == []


def test_returns_empty_when_all_rankings_empty() -> None:
    assert reciprocal_rank_fusion([[], []]) == []


def test_k_smoothing_flattens_top_rank_dominance() -> None:
    # With small k the top of a ranking dominates; with huge k the
    # contribution per position evens out, so the second-ranked-but-
    # double-appearing item should win.
    rankings = [["a", "b"], ["c", "b"]]

    small_k = dict(reciprocal_rank_fusion(rankings, k=0))
    large_k = dict(reciprocal_rank_fusion(rankings, k=10_000))

    # At k=0, 'a' (rank 1 in one list, score=1) ties with 'c' (rank 1 in
    # the other) and 'b' has 1/2 + 1/2 = 1, also tied. So all three are
    # close in small_k; in large_k, 'b' wins decisively.
    assert large_k["b"] > large_k["a"]
    assert large_k["b"] > large_k["c"]
    # And small_k still ranks 'b' at the top thanks to two contributions.
    assert small_k["b"] >= small_k["a"]
    assert small_k["b"] >= small_k["c"]


def test_works_with_integer_chunk_ids() -> None:
    fused = reciprocal_rank_fusion([[10, 20, 30], [30, 20]])
    top_id, _ = fused[0]
    assert top_id == 30


def test_rejects_negative_k() -> None:
    with pytest.raises(ValueError, match="k must be non-negative"):
        reciprocal_rank_fusion([["a"]], k=-1)
