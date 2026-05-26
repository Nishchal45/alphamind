"""Price-data adapter for the backtest harness.

Defines the :class:`PriceSource` Protocol every price provider has to
satisfy, the production :class:`YahooPriceSource` (yfinance + a CSV
disk cache), and the helpers that pick out entry / exit prices and
compute simple returns over a holding window.

Tests inject a stub :class:`PriceSource`; nothing under ``pytest``
touches the network.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd

logger = logging.getLogger(__name__)


class PriceSource(Protocol):
    """Pluggable contract for fetching adjusted-close price series.

    Implementations return a ``pandas.Series`` indexed by date,
    covering ``[start, end]`` inclusive, with rows for trading days
    only. The series may be empty if no data is available for the
    ticker / window pair.
    """

    def get_window(self, ticker: str, start: date, end: date) -> pd.Series: ...


class PriceLookupError(RuntimeError):
    """Raised when entry or exit price can't be resolved."""


class YahooPriceSource:
    """Production :class:`PriceSource` backed by yfinance with a CSV cache.

    First call hits Yahoo; the response is cached to
    ``{cache_dir}/{ticker}_{start}_{end}.csv``. Subsequent calls with
    the same window are pure disk reads. Tests stub the protocol
    instead of mocking yfinance; this class is only exercised
    end-to-end through the ``make backtest`` path.
    """

    def __init__(self, *, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    def get_window(self, ticker: str, start: date, end: date) -> pd.Series:
        if end < start:
            raise ValueError(f"end {end} predates start {start}")

        cache_path = self._cache_path(ticker, start, end)
        if cache_path.exists():
            logger.debug("price cache hit: %s", cache_path)
            return self._read_cached(cache_path)

        logger.info("fetching prices: %s %s..%s", ticker, start, end)
        import yfinance as yf  # noqa: PLC0415 — heavy import, lazy

        # yfinance's ``end`` is exclusive; nudge by one day so the
        # caller's inclusive contract holds.
        raw = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=True,
            actions=False,
        )

        if raw is None or raw.empty:
            logger.warning("no price data for %s %s..%s", ticker, start, end)
            empty: pd.Series = pd.Series(dtype="float64", name=ticker)
            self._write_cached(cache_path, empty)
            return empty

        # yfinance returns a MultiIndex column DataFrame when only one
        # ticker is requested in some versions; flatten to "Close".
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.rename(ticker).astype("float64")
        # The index is a DatetimeIndex; normalise to plain dates so
        # the cache survives round-tripping through CSV cleanly.
        close.index = pd.to_datetime(close.index).date
        self._write_cached(cache_path, close)
        return close

    def _cache_path(self, ticker: str, start: date, end: date) -> Path:
        safe_ticker = ticker.replace("/", "_")
        return self._cache_dir / f"{safe_ticker}_{start.isoformat()}_{end.isoformat()}.csv"

    @staticmethod
    def _read_cached(path: Path) -> pd.Series:
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = df["date"].dt.date
        return df.set_index("date")["close"].astype("float64")

    @staticmethod
    def _write_cached(path: Path, series: pd.Series) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = series.rename("close").to_frame()
        df.index.name = "date"
        df.to_csv(path)


def entry_price(
    source: PriceSource,
    *,
    ticker: str,
    as_of: date,
    lookahead_days: int = 7,
) -> tuple[date, float]:
    """Return ``(date, close)`` for the first trading day strictly after ``as_of``.

    Uses a small look-ahead window so weekends and holidays don't
    fail the lookup. If no trading day appears in
    ``(as_of, as_of + lookahead_days]``, raises
    :class:`PriceLookupError` — the caller decides whether to mark
    the case as errored or to widen the window.

    The strict ``>`` is the lookahead defense from ADR 0010: the
    as-of-day's close is not used, because that's the day the
    underlying filing became public and a clean backtest shouldn't
    let the test position act on the market's reaction to that
    filing.
    """
    series = source.get_window(ticker, as_of, as_of + timedelta(days=lookahead_days))
    forward = series[series.index > as_of]
    if forward.empty:
        raise PriceLookupError(
            f"no trading day strictly after {as_of} within {lookahead_days}d for {ticker}"
        )
    entry_dt = forward.index[0]
    return entry_dt, float(forward.iloc[0])


def exit_price(
    source: PriceSource,
    *,
    ticker: str,
    as_of: date,
    horizon_days: int,
    lookback_days: int = 7,
) -> tuple[date, float]:
    """Return ``(date, close)`` for the last trading day on or before ``as_of + horizon_days``.

    The lookback handles weekends / holidays at the right edge — the
    holding period is approximate by construction.
    """
    target = as_of + timedelta(days=horizon_days)
    series = source.get_window(ticker, target - timedelta(days=lookback_days), target)
    eligible = series[series.index <= target]
    if eligible.empty:
        raise PriceLookupError(
            f"no trading day on or before {target} within {lookback_days}d for {ticker}"
        )
    exit_dt = eligible.index[-1]
    return exit_dt, float(eligible.iloc[-1])


def simple_return(entry: float, exit_: float) -> float:
    """Return on a long position; sign-flip for shorts."""
    if entry <= 0:
        raise ValueError(f"non-positive entry price {entry}")
    return (exit_ - entry) / entry


__all__ = [
    "PriceLookupError",
    "PriceSource",
    "YahooPriceSource",
    "entry_price",
    "exit_price",
    "simple_return",
]
