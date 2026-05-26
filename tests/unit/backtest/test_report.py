"""Tests for the report writers.

The chart writer is exercised only smoke-style (file is non-empty
PNG); matplotlib output diffing is out of scope.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from alphamind.backtest.report import (
    CAVEAT_HEADER,
    write_chart,
    write_json,
    write_markdown,
)
from alphamind.backtest.types import (
    BacktestCase,
    BacktestReport,
    BacktestUniverse,
    CaseResult,
    Signal,
    Summary,
)


def _make_report() -> BacktestReport:
    universe = BacktestUniverse(
        cases=(
            BacktestCase(
                case_id="nvda",
                ticker="NVDA",
                query="q",
                as_of=date(2024, 1, 1),
            ),
        ),
        horizon_days=90,
        signal_threshold=0.2,
        benchmark_ticker="SPY",
        top_k=8,
    )
    result = CaseResult(
        case_id="nvda",
        ticker="NVDA",
        as_of=date(2024, 1, 1),
        entry_date=date(2024, 1, 2),
        exit_date=date(2024, 3, 29),
        entry_price=100.0,
        exit_price=110.0,
        signal=Signal(score=1.0, position="long", n_bull=3, n_bear=0),
        position_return=0.10,
        spy_return=0.04,
        alpha=0.06,
        correct=True,
    )
    summary = Summary(
        n_cases=1,
        n_active=1,
        n_errors=0,
        hit_rate=1.0,
        mean_alpha=0.06,
        cagr=None,
        sharpe=None,
        max_drawdown=0.0,
    )
    return BacktestReport(universe=universe, results=(result,), summary=summary)


def test_write_markdown_includes_caveats_and_per_case_row(tmp_path: Path) -> None:
    report = _make_report()
    out = tmp_path / "backtest.md"
    write_markdown(report, out)
    text = out.read_text()
    # Caveats first.
    assert CAVEAT_HEADER.strip() in text
    # Aggregate metrics row.
    assert "hit_rate" in text
    # Per-case row with ticker and position.
    assert "| nvda |" in text
    assert "NVDA" in text
    assert "long" in text


def test_write_json_round_trips(tmp_path: Path) -> None:
    report = _make_report()
    out = tmp_path / "backtest.json"
    write_json(report, out)
    payload = json.loads(out.read_text())
    assert payload["summary"]["hit_rate"] == 1.0
    assert payload["results"][0]["ticker"] == "NVDA"
    assert payload["results"][0]["as_of"] == "2024-01-01"
    assert payload["universe"]["benchmark_ticker"] == "SPY"


def test_write_chart_produces_nonempty_file(tmp_path: Path) -> None:
    report = _make_report()
    out = tmp_path / "chart.png"
    write_chart(report, out)
    assert out.exists()
    # PNG files start with the 8-byte PNG signature.
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_write_chart_is_a_noop_when_no_active_cases(tmp_path: Path) -> None:
    """A report with only flat / errored cases produces no curve, no PNG."""
    flat_signal = Signal(score=0.0, position="flat", n_bull=0, n_bear=0)
    result = CaseResult(
        case_id="flat",
        ticker="X",
        as_of=date(2024, 1, 1),
        entry_date=None,
        exit_date=None,
        entry_price=None,
        exit_price=None,
        signal=flat_signal,
        position_return=None,
        spy_return=None,
        alpha=None,
        correct=None,
    )
    universe = BacktestUniverse(
        cases=(BacktestCase(case_id="flat", ticker="X", query="q", as_of=date(2024, 1, 1)),),
        horizon_days=90,
        signal_threshold=0.2,
        benchmark_ticker="SPY",
        top_k=8,
    )
    report = BacktestReport(universe=universe, results=(result,), summary=None)
    out = tmp_path / "chart.png"
    write_chart(report, out)
    assert not out.exists()
