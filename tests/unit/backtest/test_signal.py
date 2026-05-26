"""Tests for the mechanical thesis-to-signal extractor."""

from __future__ import annotations

import pytest

from alphamind.agents.state import Thesis, ThesisClaim
from alphamind.backtest.signal import DEFAULT_THRESHOLD, extract_signal


def _thesis(*, bull: int, bear: int) -> Thesis:
    """Build a thesis with ``bull`` bull claims and ``bear`` bear claims."""
    bull_case = tuple(
        ThesisClaim(claim=f"bull #{i}", cited_chunk_ids=(i,)) for i in range(bull)
    )
    bear_case = tuple(
        ThesisClaim(claim=f"bear #{i}", cited_chunk_ids=(100 + i,)) for i in range(bear)
    )
    return Thesis(summary="x", bull_case=bull_case, bear_case=bear_case)


def test_balanced_thesis_is_flat() -> None:
    sig = extract_signal(_thesis(bull=2, bear=2))
    assert sig.score == 0.0
    assert sig.position == "flat"
    assert sig.n_bull == 2
    assert sig.n_bear == 2


def test_all_bull_is_long_with_score_one() -> None:
    sig = extract_signal(_thesis(bull=3, bear=0))
    assert sig.score == 1.0
    assert sig.position == "long"


def test_all_bear_is_short_with_score_minus_one() -> None:
    sig = extract_signal(_thesis(bull=0, bear=3))
    assert sig.score == -1.0
    assert sig.position == "short"


def test_score_above_threshold_goes_long() -> None:
    # 3 bull / 1 bear → score = 0.5, comfortably above 0.20.
    sig = extract_signal(_thesis(bull=3, bear=1))
    assert sig.position == "long"
    assert sig.score == 0.5


def test_score_below_threshold_stays_flat() -> None:
    # 3 bull / 2 bear → score = 0.2, exactly the default threshold.
    # The comparator is strict ``>``, so this lands flat.
    sig = extract_signal(_thesis(bull=3, bear=2))
    assert sig.position == "flat"
    assert sig.score == pytest.approx(0.2)


def test_empty_thesis_is_flat() -> None:
    sig = extract_signal(_thesis(bull=0, bear=0))
    assert sig.score == 0.0
    assert sig.position == "flat"


def test_none_thesis_is_flat() -> None:
    sig = extract_signal(None)
    assert sig.score == 0.0
    assert sig.position == "flat"
    assert sig.n_bull == 0
    assert sig.n_bear == 0


def test_custom_threshold_widens_flat_band() -> None:
    # With threshold = 0.6, a 3 / 1 thesis (score 0.5) stays flat.
    sig = extract_signal(_thesis(bull=3, bear=1), threshold=0.6)
    assert sig.position == "flat"


def test_threshold_zero_makes_every_imbalance_active() -> None:
    sig = extract_signal(_thesis(bull=2, bear=1), threshold=0.0)
    assert sig.position == "long"


def test_negative_threshold_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        extract_signal(_thesis(bull=1, bear=0), threshold=-0.1)


def test_default_threshold_value_is_documented() -> None:
    # If this changes, update ADR 0010 too.
    assert DEFAULT_THRESHOLD == 0.20
