"""End-to-end tests for the backtest runner.

Wires a stub graph and a stub :class:`PriceSource` so the full
run-one-case flow can be exercised without LLMs, Postgres, or the
network. Each test pins one branch of the runner: happy long, happy
short, flat signal, graph failure, price-lookup failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd
import pytest

from alphamind.agents.state import Thesis, ThesisClaim
from alphamind.backtest.loader import DEFAULT_SIGNAL_THRESHOLD
from alphamind.backtest.runner import run_backtest
from alphamind.backtest.types import BacktestCase, BacktestUniverse

pytestmark = pytest.mark.asyncio


@dataclass
class StubGraph:
    """Routes by query substring to a canned state dict."""

    responses: dict[str, dict[str, Any]] = field(default_factory=dict)
    raise_for: set[str] = field(default_factory=set)
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def ainvoke(self, state_input: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(state_input))
        query = state_input["query"]
        for marker in self.raise_for:
            if marker in query:
                raise RuntimeError(f"stub graph blew up on marker {marker!r}")
        for marker, state in self.responses.items():
            if marker in query:
                return state
        return {}  # no thesis → flat


@dataclass
class StubPriceSource:
    """Returns a canned pandas Series regardless of the requested window."""

    series_by_ticker: dict[str, pd.Series]
    calls: list[tuple[str, date, date]] = field(default_factory=list)

    def get_window(self, ticker: str, start: date, end: date) -> pd.Series:
        self.calls.append((ticker, start, end))
        series = self.series_by_ticker.get(ticker, pd.Series(dtype="float64"))
        if series.empty:
            return series
        mask = (series.index >= start) & (series.index <= end)
        return series[mask]


def _prices(values: dict[date, float]) -> pd.Series:
    index = sorted(values.keys())
    return pd.Series([values[d] for d in index], index=index, dtype="float64")


def _bullish_thesis() -> Thesis:
    return Thesis(
        summary="x",
        bull_case=(
            ThesisClaim(claim="b1", cited_chunk_ids=(1,)),
            ThesisClaim(claim="b2", cited_chunk_ids=(2,)),
            ThesisClaim(claim="b3", cited_chunk_ids=(3,)),
        ),
        bear_case=(),
    )


def _bearish_thesis() -> Thesis:
    return Thesis(
        summary="x",
        bull_case=(),
        bear_case=(
            ThesisClaim(claim="r1", cited_chunk_ids=(1,)),
            ThesisClaim(claim="r2", cited_chunk_ids=(2,)),
            ThesisClaim(claim="r3", cited_chunk_ids=(3,)),
        ),
    )


def _flat_thesis() -> Thesis:
    return Thesis(
        summary="x",
        bull_case=(ThesisClaim(claim="b", cited_chunk_ids=(1,)),),
        bear_case=(ThesisClaim(claim="r", cited_chunk_ids=(2,)),),
    )


def _universe(cases: list[BacktestCase]) -> BacktestUniverse:
    return BacktestUniverse(
        cases=tuple(cases),
        horizon_days=90,
        signal_threshold=DEFAULT_SIGNAL_THRESHOLD,
        benchmark_ticker="SPY",
        top_k=8,
    )


async def test_long_case_happy_path() -> None:
    case = BacktestCase(
        case_id="nvda-bull",
        ticker="NVDA",
        query="bull/bear on NVDA",
        as_of=date(2024, 1, 1),
    )
    graph = StubGraph(responses={"NVDA": {"thesis": _bullish_thesis()}})
    prices = StubPriceSource(
        series_by_ticker={
            "NVDA": _prices(
                {
                    date(2024, 1, 2): 100.0,
                    date(2024, 3, 29): 120.0,  # +20% over hold (target = 2024-03-31)
                }
            ),
            "SPY": _prices(
                {
                    date(2024, 1, 2): 400.0,
                    date(2024, 3, 29): 420.0,  # +5%
                }
            ),
        }
    )

    report = await run_backtest(_universe([case]), graph=graph, prices=prices)  # type: ignore[arg-type]
    assert len(report.results) == 1
    r = report.results[0]
    assert r.signal.position == "long"
    assert r.error is None
    assert r.position_return == pytest.approx(0.20)
    assert r.spy_return == pytest.approx(0.05)
    assert r.alpha == pytest.approx(0.15)
    assert r.correct is True


async def test_short_case_correctness_flips_with_negative_return() -> None:
    case = BacktestCase(
        case_id="tsla-bear",
        ticker="TSLA",
        query="bull/bear on TSLA",
        as_of=date(2024, 1, 1),
    )
    graph = StubGraph(responses={"TSLA": {"thesis": _bearish_thesis()}})
    prices = StubPriceSource(
        series_by_ticker={
            "TSLA": _prices(
                {
                    date(2024, 1, 2): 100.0,
                    date(2024, 3, 29): 80.0,  # raw -20% (target = 2024-03-31)
                }
            ),
            "SPY": _prices(
                {
                    date(2024, 1, 2): 400.0,
                    date(2024, 3, 29): 408.0,  # +2%
                }
            ),
        }
    )

    report = await run_backtest(_universe([case]), graph=graph, prices=prices)  # type: ignore[arg-type]
    r = report.results[0]
    assert r.signal.position == "short"
    assert r.position_return == pytest.approx(-0.20)  # raw, signed by ticker
    # Position-aware: short with -20% raw move is +20% portfolio; correct.
    assert r.correct is True
    # Alpha for short: +20% portfolio minus +2% SPY = +18%.
    assert r.alpha == pytest.approx(0.18)


async def test_flat_signal_skips_price_lookup() -> None:
    case = BacktestCase(
        case_id="msft-flat",
        ticker="MSFT",
        query="anything on MSFT",
        as_of=date(2024, 1, 1),
    )
    graph = StubGraph(responses={"MSFT": {"thesis": _flat_thesis()}})
    prices = StubPriceSource(series_by_ticker={})  # would raise if consulted

    report = await run_backtest(_universe([case]), graph=graph, prices=prices)  # type: ignore[arg-type]
    r = report.results[0]
    assert r.signal.position == "flat"
    assert r.entry_price is None
    assert r.exit_price is None
    assert r.position_return is None
    assert r.alpha is None
    assert r.correct is None
    assert prices.calls == []  # short-circuit before price lookups


async def test_graph_failure_is_isolated_to_one_case() -> None:
    cases = [
        BacktestCase(case_id="ok", ticker="NVDA", query="NVDA", as_of=date(2024, 1, 1)),
        BacktestCase(case_id="bad", ticker="AAPL", query="AAPL", as_of=date(2024, 2, 1)),
    ]
    graph = StubGraph(
        responses={"NVDA": {"thesis": _bullish_thesis()}},
        raise_for={"AAPL"},
    )
    prices = StubPriceSource(
        series_by_ticker={
            "NVDA": _prices({date(2024, 1, 2): 100.0, date(2024, 3, 29): 110.0}),
            "SPY": _prices({date(2024, 1, 2): 400.0, date(2024, 3, 29): 404.0}),
        }
    )

    report = await run_backtest(_universe(cases), graph=graph, prices=prices)  # type: ignore[arg-type]
    assert len(report.results) == 2
    ok_result = next(r for r in report.results if r.case_id == "ok")
    bad_result = next(r for r in report.results if r.case_id == "bad")
    assert ok_result.error is None
    assert bad_result.error is not None
    assert "graph" in bad_result.error
    # Summary reflects the split.
    assert report.summary is not None
    assert report.summary.n_active == 1
    assert report.summary.n_errors == 1


async def test_missing_prices_marks_case_errored() -> None:
    case = BacktestCase(
        case_id="nvda",
        ticker="NVDA",
        query="NVDA",
        as_of=date(2024, 1, 1),
    )
    graph = StubGraph(responses={"NVDA": {"thesis": _bullish_thesis()}})
    # NVDA has no forward data; SPY has plenty.
    prices = StubPriceSource(
        series_by_ticker={
            "NVDA": _prices({date(2024, 1, 1): 100.0}),  # no day strictly after
            "SPY": _prices({date(2024, 1, 2): 400.0, date(2024, 4, 1): 420.0}),
        }
    )

    report = await run_backtest(_universe([case]), graph=graph, prices=prices)  # type: ignore[arg-type]
    r = report.results[0]
    assert r.error is not None
    assert "prices" in r.error
    assert r.position_return is None
    assert r.correct is None
