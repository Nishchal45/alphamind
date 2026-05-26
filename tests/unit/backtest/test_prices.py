"""Tests for the price-lookup helpers.

Everything is exercised against a stub :class:`PriceSource` that
returns a hand-crafted series. The production yfinance path is not
exercised in unit tests — it's an end-to-end integration concern,
covered by the ``make backtest`` smoke test in CI's manual lane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import pytest

from alphamind.backtest.prices import (
    PriceLookupError,
    entry_price,
    exit_price,
    simple_return,
)


@dataclass
class StubPriceSource:
    """Returns a canned pandas Series regardless of the requested window."""

    series: pd.Series
    calls: list[tuple[str, date, date]] = field(default_factory=list)

    def get_window(self, ticker: str, start: date, end: date) -> pd.Series:
        self.calls.append((ticker, start, end))
        mask = (self.series.index >= start) & (self.series.index <= end)
        return self.series[mask]


def _series(values: dict[date, float]) -> pd.Series:
    """Build a date-indexed close series from a date→price dict."""
    index = sorted(values.keys())
    return pd.Series([values[d] for d in index], index=index, dtype="float64")


def test_entry_price_picks_first_trading_day_after_as_of() -> None:
    series = _series(
        {
            date(2024, 1, 2): 100.0,
            date(2024, 1, 3): 101.0,
            date(2024, 1, 4): 102.0,
        }
    )
    src = StubPriceSource(series=series)
    dt, price = entry_price(src, ticker="NVDA", as_of=date(2024, 1, 2))
    # as_of-day price is NOT used — strict ``>``.
    assert dt == date(2024, 1, 3)
    assert price == 101.0


def test_entry_price_skips_weekend_gap() -> None:
    # as_of = Friday; entry should be the following Monday.
    series = _series(
        {
            date(2024, 1, 5): 100.0,  # Friday
            date(2024, 1, 8): 105.0,  # Monday
        }
    )
    src = StubPriceSource(series=series)
    dt, price = entry_price(src, ticker="NVDA", as_of=date(2024, 1, 5))
    assert dt == date(2024, 1, 8)
    assert price == 105.0


def test_entry_price_raises_when_no_forward_data() -> None:
    series = _series({date(2024, 1, 2): 100.0})
    src = StubPriceSource(series=series)
    with pytest.raises(PriceLookupError, match="strictly after"):
        entry_price(src, ticker="NVDA", as_of=date(2024, 1, 2))


def test_exit_price_picks_last_trading_day_on_or_before_target() -> None:
    series = _series(
        {
            date(2024, 1, 2): 100.0,
            date(2024, 4, 1): 110.0,  # Monday
            date(2024, 4, 2): 111.0,  # exit target
            date(2024, 4, 3): 112.0,  # after target
        }
    )
    src = StubPriceSource(series=series)
    dt, price = exit_price(
        src,
        ticker="NVDA",
        as_of=date(2024, 1, 2),
        horizon_days=91,
    )
    # target = 2024-04-02; eligible up to and including that day.
    assert dt == date(2024, 4, 2)
    assert price == 111.0


def test_exit_price_falls_back_to_earlier_trading_day_when_target_is_weekend() -> None:
    # horizon target lands on Sunday; should pick the prior Friday.
    series = _series(
        {
            date(2024, 1, 5): 100.0,  # Friday
        }
    )
    src = StubPriceSource(series=series)
    dt, price = exit_price(
        src,
        ticker="NVDA",
        as_of=date(2024, 1, 1),
        horizon_days=6,  # target = Sunday 2024-01-07
    )
    assert dt == date(2024, 1, 5)
    assert price == 100.0


def test_exit_price_raises_when_no_data_in_window() -> None:
    series = _series({date(2025, 1, 1): 100.0})  # far outside target window
    src = StubPriceSource(series=series)
    with pytest.raises(PriceLookupError, match="on or before"):
        exit_price(src, ticker="NVDA", as_of=date(2024, 1, 1), horizon_days=90)


def test_simple_return() -> None:
    assert simple_return(100.0, 110.0) == pytest.approx(0.10)
    assert simple_return(100.0, 90.0) == pytest.approx(-0.10)


def test_simple_return_rejects_non_positive_entry() -> None:
    with pytest.raises(ValueError, match="non-positive entry"):
        simple_return(0.0, 10.0)
