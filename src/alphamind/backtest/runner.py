"""Backtest runner.

Walks a :class:`BacktestUniverse`, runs the compiled agent graph at
each historical as-of date, extracts a signal from the resulting
thesis, and computes forward returns + alpha vs. the benchmark.

The runner accepts a pre-built compiled graph and a
:class:`PriceSource`, so tests inject scripted stubs and exercise the
full pipeline end-to-end without Postgres, the LLM, or yfinance.

Per-case exceptions are isolated: one bad lookup does not abort the
backtest. The errored case lands in the report with an ``error``
field and is excluded from active-only aggregates.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from alphamind.agents.state import Thesis
from alphamind.backtest.metrics import summarise
from alphamind.backtest.prices import (
    PriceLookupError,
    PriceSource,
    entry_price,
    exit_price,
    simple_return,
)
from alphamind.backtest.signal import extract_signal
from alphamind.backtest.types import (
    BacktestCase,
    BacktestReport,
    BacktestUniverse,
    CaseResult,
)

logger = logging.getLogger(__name__)


async def run_backtest(
    universe: BacktestUniverse,
    *,
    graph: CompiledStateGraph[Any, Any, Any, Any],
    prices: PriceSource,
) -> BacktestReport:
    """Run every case in ``universe`` and return the aggregated report."""

    results: list[CaseResult] = []
    for case in universe.cases:
        result = await _run_one(
            case,
            graph=graph,
            prices=prices,
            horizon_days=universe.horizon_days,
            threshold=universe.signal_threshold,
            benchmark=universe.benchmark_ticker,
            top_k=universe.top_k,
        )
        results.append(result)

    return BacktestReport(
        universe=universe,
        results=tuple(results),
        summary=summarise(results),
    )


async def _run_one(
    case: BacktestCase,
    *,
    graph: CompiledStateGraph[Any, Any, Any, Any],
    prices: PriceSource,
    horizon_days: int,
    threshold: float,
    benchmark: str,
    top_k: int,
) -> CaseResult:
    """Execute one case and return its :class:`CaseResult`."""

    # 1. Run the agent graph at the historical as-of.
    try:
        state: dict[str, Any] = await graph.ainvoke(
            {
                "query": case.query,
                "as_of": case.as_of,
                "top_k": top_k,
            }
        )
    except Exception as exc:  # graph-side failure
        logger.warning("case %s: graph failure: %s", case.case_id, exc)
        return _error_result(case, threshold=threshold, error=f"graph: {exc}")

    thesis: Thesis | None = state.get("thesis")
    signal = extract_signal(thesis, threshold=threshold)

    if signal.position == "flat":
        return CaseResult(
            case_id=case.case_id,
            ticker=case.ticker,
            as_of=case.as_of,
            entry_date=None,
            exit_date=None,
            entry_price=None,
            exit_price=None,
            signal=signal,
            position_return=None,
            spy_return=None,
            alpha=None,
            correct=None,
        )

    # 2. Resolve entry / exit prices for the position ticker.
    try:
        entry_dt, entry_px = entry_price(prices, ticker=case.ticker, as_of=case.as_of)
        exit_dt, exit_px = exit_price(
            prices, ticker=case.ticker, as_of=case.as_of, horizon_days=horizon_days
        )
    except PriceLookupError as exc:
        return _error_result(case, threshold=threshold, error=f"prices: {exc}")

    # 3. Resolve the benchmark over the same window.
    try:
        _, spy_entry = entry_price(prices, ticker=benchmark, as_of=case.as_of)
        _, spy_exit = exit_price(
            prices, ticker=benchmark, as_of=case.as_of, horizon_days=horizon_days
        )
    except PriceLookupError as exc:
        return _error_result(case, threshold=threshold, error=f"benchmark: {exc}")

    raw_return = simple_return(entry_px, exit_px)
    spy_return = simple_return(spy_entry, spy_exit)

    # 4. Position-aware return drives correctness; alpha stays raw
    #    (long vs SPY, short alpha would be (short_return - (-spy_return))
    #    which is too cute — alpha here is the long-equivalent comparison).
    correct: bool
    if signal.position == "long":
        correct = raw_return > 0
        alpha = raw_return - spy_return
    else:  # short
        correct = raw_return < 0
        # For a short position, "alpha vs SPY" means: portfolio gained
        # (-raw_return), benchmark gained spy_return. Both signs flipped
        # so the report's alpha column is comparable across long / short.
        alpha = (-raw_return) - spy_return

    return CaseResult(
        case_id=case.case_id,
        ticker=case.ticker,
        as_of=case.as_of,
        entry_date=entry_dt,
        exit_date=exit_dt,
        entry_price=entry_px,
        exit_price=exit_px,
        signal=signal,
        position_return=raw_return,
        spy_return=spy_return,
        alpha=alpha,
        correct=correct,
    )


def _error_result(case: BacktestCase, *, threshold: float, error: str) -> CaseResult:
    """Build a CaseResult representing a per-case failure.

    ``threshold`` is preserved for the signal field even though the
    case never produced a thesis, so the report can show "we tried,
    we failed" rows in the same shape as successful ones.
    """
    # Mirror the flat-signal shape: this case did not produce a position.
    from alphamind.backtest.signal import extract_signal as _flat  # noqa: PLC0415

    flat_signal = _flat(None, threshold=threshold)
    return CaseResult(
        case_id=case.case_id,
        ticker=case.ticker,
        as_of=case.as_of,
        entry_date=None,
        exit_date=None,
        entry_price=None,
        exit_price=None,
        signal=flat_signal,
        position_return=None,
        spy_return=None,
        alpha=None,
        correct=None,
        error=error,
    )


__all__ = ["run_backtest"]
