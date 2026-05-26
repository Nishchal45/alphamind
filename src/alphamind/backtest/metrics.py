"""Aggregate metrics over the per-case results.

Every metric is a pure function over a sequence of :class:`CaseResult`
records. None values mean "case not active" (flat signal, or
errored); rate-style metrics average over active cases only.

The annualisation used by ``cagr`` and ``sharpe`` treats each case as
one observation and uses the calendar span between the first and
last as-of date to convert to per-year. ADR 0010 explicitly flags
these as descriptive on small samples, not statistically meaningful.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from alphamind.backtest.types import CaseResult, Summary

# Approximation used to convert case observations into a per-year
# rate. The active-window length is measured in calendar days and
# normalised to a 365.25-day year.
DAYS_PER_YEAR = 365.25

# Minimum sample size for variance-based metrics (Sharpe, span-based
# annualisation). Below this both the mean and the std-of-mean are
# meaningless; metrics short-circuit to None.
_MIN_OBSERVATIONS = 2


def position_return_signed(result: CaseResult) -> float | None:
    """Per-case return *with* the long/short sign applied.

    Long: raw return as-is. Short: sign-flipped (a -10% raw return on
    a short position is a +10% portfolio contribution). Flat or
    errored: ``None``.
    """

    if result.position_return is None or not result.has_position:
        return None
    if result.signal.position == "long":
        return result.position_return
    if result.signal.position == "short":
        return -result.position_return
    return None  # pragma: no cover — defensive; "flat" already filtered above


def equity_curve(results: Sequence[CaseResult]) -> list[tuple[str, float]]:
    """Cumulative-product equity curve over the active cases.

    Cases are ordered by ``as_of`` ascending, then by ``case_id`` to
    break ties stably. The curve starts at ``1.0`` and multiplies by
    ``(1 + signed_return)`` per active case. Returns a list of
    ``(case_id, equity)`` pairs; the empty list when no case is
    active.

    This is a portfolio narrative, not a time-aware accounting — see
    ADR 0010 for the limitation.
    """

    ordered = sorted(results, key=lambda r: (r.as_of, r.case_id))
    points: list[tuple[str, float]] = []
    equity = 1.0
    for r in ordered:
        ret = position_return_signed(r)
        if ret is None:
            continue
        equity *= 1.0 + ret
        points.append((r.case_id, equity))
    return points


def hit_rate(results: Sequence[CaseResult]) -> float | None:
    """Fraction of active cases where the position was correct."""
    active = [r for r in results if r.has_position and r.correct is not None]
    if not active:
        return None
    return sum(1 for r in active if r.correct) / len(active)


def mean_alpha(results: Sequence[CaseResult]) -> float | None:
    """Arithmetic mean of alpha over active cases."""
    active = [r.alpha for r in results if r.has_position and r.alpha is not None]
    if not active:
        return None
    return sum(active) / len(active)


def max_drawdown(curve: Sequence[tuple[str, float]]) -> float | None:
    """Worst peak-to-trough drawdown on the equity curve.

    Returned as a negative fraction (a 20% drawdown is ``-0.20``).
    None when the curve is empty.
    """
    if not curve:
        return None
    peak = curve[0][1]
    worst = 0.0
    for _, value in curve:
        peak = max(peak, value)
        drawdown = (value - peak) / peak
        worst = min(worst, drawdown)
    return worst


def _active_span_days(results: Sequence[CaseResult]) -> int:
    """Calendar days between the first and last active as-of date.

    Returns 0 when there are no active cases or only one case.
    """
    active_dates = [r.as_of for r in results if r.has_position]
    if len(active_dates) < _MIN_OBSERVATIONS:
        return 0
    return (max(active_dates) - min(active_dates)).days


def cagr(results: Sequence[CaseResult]) -> float | None:
    """Compound annual growth rate over the active span.

    Returns ``None`` if the span is zero (one or zero active cases)
    or the final equity is non-positive (a loss large enough to
    wipe the notional out).
    """
    curve = equity_curve(results)
    if not curve:
        return None
    final_equity: float = curve[-1][1]
    if final_equity <= 0:
        return None
    span_days = _active_span_days(results)
    if span_days == 0:
        return None
    years = span_days / DAYS_PER_YEAR
    return float(final_equity ** (1.0 / years)) - 1.0


def sharpe(results: Sequence[CaseResult]) -> float | None:
    """Sharpe ratio over active cases, annualised by active span.

    Risk-free rate is 0. Returns ``None`` when fewer than two active
    cases (std is undefined) or when the std is zero (degenerate).
    """
    raw_returns = [position_return_signed(r) for r in results if r.has_position]
    returns: list[float] = [v for v in raw_returns if v is not None]
    if len(returns) < _MIN_OBSERVATIONS:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return None
    span_days = _active_span_days(results)
    if span_days == 0:
        return None
    obs_per_year = len(returns) * DAYS_PER_YEAR / span_days
    return (mean / std) * math.sqrt(obs_per_year)


def summarise(results: Sequence[CaseResult]) -> Summary:
    """Roll the per-case results into the report-level :class:`Summary`."""
    return Summary(
        n_cases=len(results),
        n_active=sum(1 for r in results if r.has_position),
        n_errors=sum(1 for r in results if r.error is not None),
        hit_rate=hit_rate(results),
        mean_alpha=mean_alpha(results),
        cagr=cagr(results),
        sharpe=sharpe(results),
        max_drawdown=max_drawdown(equity_curve(results)),
    )


__all__ = [
    "cagr",
    "equity_curve",
    "hit_rate",
    "max_drawdown",
    "mean_alpha",
    "position_return_signed",
    "sharpe",
    "summarise",
]
