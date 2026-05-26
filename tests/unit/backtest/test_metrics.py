"""Tests for the aggregate metrics."""

from __future__ import annotations

from datetime import date

import pytest

from alphamind.backtest.metrics import (
    cagr,
    equity_curve,
    hit_rate,
    max_drawdown,
    mean_alpha,
    position_return_signed,
    sharpe,
    summarise,
)
from alphamind.backtest.types import CaseResult, Position, Signal


def _case(
    *,
    case_id: str = "c1",
    as_of: date = date(2024, 1, 1),
    position: Position = "long",
    position_return: float | None = 0.1,
    spy_return: float | None = 0.05,
    correct: bool | None = True,
    error: str | None = None,
) -> CaseResult:
    """Build a synthetic CaseResult with sane defaults."""
    score = 0.5 if position == "long" else -0.5
    signal = Signal(score=score, position=position, n_bull=1, n_bear=0)
    alpha = (
        position_return - spy_return
        if (position_return is not None and spy_return is not None and position != "flat")
        else None
    )
    return CaseResult(
        case_id=case_id,
        ticker="NVDA",
        as_of=as_of,
        entry_date=as_of,
        exit_date=as_of,
        entry_price=100.0,
        exit_price=100.0 * (1 + position_return) if position_return is not None else None,
        signal=signal,
        position_return=position_return,
        spy_return=spy_return,
        alpha=alpha,
        correct=correct,
        error=error,
    )


def test_position_return_signed_flips_for_short() -> None:
    long_case = _case(position="long", position_return=0.10)
    short_case = _case(position="short", position_return=-0.10)
    assert position_return_signed(long_case) == 0.10
    # short with -10% raw move = +10% portfolio contribution.
    assert position_return_signed(short_case) == 0.10


def test_position_return_signed_none_for_flat() -> None:
    flat = _case(position="flat", position_return=None, correct=None)
    assert position_return_signed(flat) is None


def test_position_return_signed_none_for_errored_case() -> None:
    errored = _case(position="long", position_return=None, error="lookup failed", correct=None)
    assert position_return_signed(errored) is None


def test_equity_curve_compounds_active_cases_in_chronological_order() -> None:
    cases = [
        _case(case_id="b", as_of=date(2024, 6, 1), position_return=0.20),
        _case(case_id="a", as_of=date(2024, 1, 1), position_return=0.10),
    ]
    curve = equity_curve(cases)
    # Ordered by as_of: a (+10%) then b (+20%) → 1.0 * 1.1 * 1.2 = 1.32.
    assert [pt[0] for pt in curve] == ["a", "b"]
    assert curve[-1][1] == pytest.approx(1.32)


def test_equity_curve_skips_flat_and_errored_cases() -> None:
    cases = [
        _case(case_id="active", position_return=0.10),
        _case(case_id="flat", position="flat", position_return=None, correct=None),
        _case(case_id="err", position_return=None, error="x", correct=None),
    ]
    curve = equity_curve(cases)
    assert [pt[0] for pt in curve] == ["active"]


def test_hit_rate_active_only() -> None:
    cases = [
        _case(case_id="a", correct=True),
        _case(case_id="b", correct=False),
        _case(case_id="c", correct=True),
        _case(case_id="flat", position="flat", position_return=None, correct=None),
    ]
    # 2/3 of active cases correct.
    assert hit_rate(cases) == pytest.approx(2 / 3)


def test_hit_rate_none_when_no_active_cases() -> None:
    assert hit_rate([]) is None
    assert (
        hit_rate(
            [_case(position="flat", position_return=None, correct=None)]
        )
        is None
    )


def test_mean_alpha_arithmetic() -> None:
    cases = [
        _case(case_id="a", position_return=0.10, spy_return=0.05),  # alpha = +5
        _case(case_id="b", position_return=0.00, spy_return=0.05),  # alpha = -5
        _case(case_id="flat", position="flat", position_return=None, correct=None),
    ]
    assert mean_alpha(cases) == pytest.approx(0.0)


def test_max_drawdown_simple() -> None:
    # Curve: 1.0 → 1.20 (peak) → 0.96 (trough) → 1.10
    # Drawdown from peak 1.20 to trough 0.96 = -20%.
    curve = [("a", 1.20), ("b", 0.96), ("c", 1.10)]
    assert max_drawdown(curve) == pytest.approx(-0.20)


def test_max_drawdown_none_on_empty_curve() -> None:
    assert max_drawdown([]) is None


def test_max_drawdown_zero_when_monotonically_increasing() -> None:
    assert max_drawdown([("a", 1.0), ("b", 1.1), ("c", 1.2)]) == 0.0


def test_cagr_annualises_over_active_span() -> None:
    # Two cases ~365 days apart, +10% then +10% → 1.21 final equity over ~1 year.
    cases = [
        _case(case_id="a", as_of=date(2024, 1, 1), position_return=0.10),
        _case(case_id="b", as_of=date(2024, 12, 31), position_return=0.10),
    ]
    result = cagr(cases)
    assert result is not None
    # ~21% CAGR (slightly off 1.0y because DAYS_PER_YEAR = 365.25).
    assert result == pytest.approx(0.21, abs=0.005)


def test_cagr_none_when_fewer_than_two_active_cases() -> None:
    assert cagr([]) is None
    assert cagr([_case(case_id="a", position_return=0.10)]) is None


def test_sharpe_none_when_fewer_than_two_returns() -> None:
    assert sharpe([_case(case_id="a", position_return=0.10)]) is None


def test_sharpe_none_when_returns_are_identical() -> None:
    cases = [
        _case(case_id="a", as_of=date(2024, 1, 1), position_return=0.10),
        _case(case_id="b", as_of=date(2024, 6, 1), position_return=0.10),
    ]
    # std = 0 → Sharpe is undefined.
    assert sharpe(cases) is None


def test_sharpe_positive_for_consistent_winners() -> None:
    cases = [
        _case(case_id=f"c{i}", as_of=date(2024, i + 1, 1), position_return=0.05 + 0.01 * i)
        for i in range(4)
    ]
    s = sharpe(cases)
    assert s is not None and s > 0


def test_summarise_aggregates_everything() -> None:
    cases = [
        _case(case_id="a", as_of=date(2024, 1, 1), correct=True),
        _case(case_id="b", as_of=date(2024, 4, 1), correct=False),
        _case(case_id="flat", position="flat", position_return=None, correct=None),
        _case(case_id="err", position_return=None, error="lookup", correct=None),
    ]
    summary = summarise(cases)
    assert summary.n_cases == 4
    assert summary.n_active == 2
    assert summary.n_errors == 1
    assert summary.hit_rate == pytest.approx(0.5)
    # Other metrics exist but their exact values aren't the point of this test.
