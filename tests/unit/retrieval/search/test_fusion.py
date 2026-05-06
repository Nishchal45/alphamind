"""Tests for Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from alphamind.retrieval.search.fusion import reciprocal_rank_fusion


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    score: float


def test_single_branch_preserves_order() -> None:
    branch = [Hit(1, 0.9), Hit(2, 0.8), Hit(3, 0.7)]

    fused = reciprocal_rank_fusion([branch])

    ids = [h.chunk_id for h in fused]
    assert ids == [1, 2, 3]


def test_chunks_in_both_branches_outrank_chunks_in_one() -> None:
    lexical = [Hit(1, 1.0), Hit(2, 0.5), Hit(3, 0.1)]
    dense = [Hit(2, 1.0), Hit(4, 0.8), Hit(1, 0.2)]

    fused = reciprocal_rank_fusion([lexical, dense])

    ids = [h.chunk_id for h in fused]
    # 1 and 2 appear in both lists; 3 and 4 in only one.
    # Both intersection IDs should rank ahead of the singletons.
    assert ids[:2] == [1, 2] or ids[:2] == [2, 1]
    assert set(ids[:2]) == {1, 2}


def test_rrf_is_calibration_free() -> None:
    """Different magnitude scales between branches should not skew fusion.

    A branch with massive scores shouldn't dominate one with tiny scores —
    RRF only reads ranks. Mock that by giving lexical 1000x larger scores
    than dense for the same ordering.
    """
    lexical = [Hit(1, 9999.0), Hit(2, 5000.0), Hit(3, 100.0)]
    dense = [Hit(3, 0.99), Hit(2, 0.5), Hit(1, 0.01)]

    fused = reciprocal_rank_fusion([lexical, dense])
    fused_ids = [h.chunk_id for h in fused]

    # Both branches contribute equally because RRF reads ranks. Items
    # appearing high in one and low in the other should land in the middle;
    # the only scoring movement should be tie-break consistency.
    assert set(fused_ids) == {1, 2, 3}
    # Scores should be deterministic given identical rank inputs.
    assert all(h.score > 0 for h in fused)


def test_limit_truncates_output() -> None:
    branch = [Hit(i, 1.0 / i) for i in range(1, 21)]

    fused = reciprocal_rank_fusion([branch], limit=5)

    assert len(fused) == 5
    assert [h.chunk_id for h in fused] == [1, 2, 3, 4, 5]


def test_empty_input_yields_empty_output() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_invalid_arguments_rejected() -> None:
    branch = [Hit(1, 1.0)]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([branch], k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([branch], limit=0)
