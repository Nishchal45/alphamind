"""Typed data shapes for the backtest harness.

Mirrors the dataclass conventions used elsewhere in the codebase
(:mod:`alphamind.agents.state`, :mod:`alphamind.eval.types`):
``frozen=True, slots=True`` for everything that flows through the
runner, and Literal types for any enum-shaped field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

Position = Literal["long", "short", "flat"]


@dataclass(frozen=True, slots=True)
class BacktestCase:
    """One row in the backtest universe.

    ``case_id`` is the unique key used in the report and in JSON
    output; ``query`` is the research question handed to the agent
    graph for this ticker / as-of pair.
    """

    case_id: str
    ticker: str
    query: str
    as_of: date


@dataclass(frozen=True, slots=True)
class BacktestUniverse:
    """The parsed YAML universe — a list of cases plus run-level config."""

    cases: tuple[BacktestCase, ...]
    horizon_days: int
    signal_threshold: float
    benchmark_ticker: str
    top_k: int


@dataclass(frozen=True, slots=True)
class Signal:
    """The mechanical signal a thesis produces.

    ``score`` lives in ``[-1, 1]`` (claim-count ratio).
    ``position`` is the thresholded result that drives portfolio
    construction. ``n_bull`` / ``n_bear`` are surfaced so the report
    can show *why* a case was rated the way it was without re-reading
    the thesis.
    """

    score: float
    position: Position
    n_bull: int
    n_bear: int


@dataclass(frozen=True, slots=True)
class CaseResult:
    """End-to-end outcome for one backtest case."""

    case_id: str
    ticker: str
    as_of: date
    entry_date: date | None
    exit_date: date | None
    entry_price: float | None
    exit_price: float | None
    signal: Signal
    position_return: float | None
    spy_return: float | None
    alpha: float | None
    correct: bool | None
    error: str | None = None

    @property
    def has_position(self) -> bool:
        """True when this case took a long or short position."""
        return self.signal.position != "flat" and self.error is None


@dataclass(frozen=True, slots=True)
class Summary:
    """Aggregate metrics across all cases.

    ``n_active`` is the count of cases that took a position; the
    rate-style metrics (``hit_rate``, ``mean_alpha``) average over
    those only — flat cases contribute nothing by definition.
    Portfolio metrics (``cagr``, ``sharpe``, ``max_drawdown``) come
    from the equal-weight equity curve across active cases.
    """

    n_cases: int
    n_active: int
    n_errors: int
    hit_rate: float | None
    mean_alpha: float | None
    cagr: float | None
    sharpe: float | None
    max_drawdown: float | None


@dataclass(frozen=True, slots=True)
class BacktestReport:
    """What the runner returns and the writer serialises."""

    universe: BacktestUniverse
    results: tuple[CaseResult, ...] = field(default_factory=tuple)
    summary: Summary | None = None


__all__ = [
    "BacktestCase",
    "BacktestReport",
    "BacktestUniverse",
    "CaseResult",
    "Position",
    "Signal",
    "Summary",
]
