"""Backtest harness.

Walks a YAML universe of ``(ticker, query, as_of)`` cases, runs the
production research DAG at each historical as-of date, extracts a
mechanical signal from each thesis, and compares the resulting
position to forward price action plus an SPY benchmark.

See ADR 0010 for the design decisions — signal extraction, lookahead
defenses, what this harness deliberately doesn't measure.
"""

from __future__ import annotations

from alphamind.backtest.types import (
    BacktestCase,
    BacktestReport,
    BacktestUniverse,
    CaseResult,
    Position,
    Signal,
    Summary,
)

__all__ = [
    "BacktestCase",
    "BacktestReport",
    "BacktestUniverse",
    "CaseResult",
    "Position",
    "Signal",
    "Summary",
]
