"""Tests for the backtest universe YAML loader."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from alphamind.backtest.loader import (
    DEFAULT_BENCHMARK,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_SIGNAL_THRESHOLD,
    UniverseError,
    load_universe,
)


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "universe.yaml"
    path.write_text(body)
    return path


def test_minimal_universe(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
cases:
  - case_id: nvda-q4-2024
    ticker: nvda
    query: What is NVDA's bull / bear case?
    as_of: 2024-12-31
""",
    )
    universe = load_universe(path)
    assert universe.horizon_days == DEFAULT_HORIZON_DAYS
    assert universe.signal_threshold == DEFAULT_SIGNAL_THRESHOLD
    assert universe.benchmark_ticker == DEFAULT_BENCHMARK
    assert len(universe.cases) == 1
    case = universe.cases[0]
    assert case.case_id == "nvda-q4-2024"
    assert case.ticker == "NVDA"  # upper-cased
    assert case.as_of == date(2024, 12, 31)


def test_overrides_apply(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
horizon_days: 30
signal_threshold: 0.4
benchmark_ticker: QQQ
top_k: 12
cases:
  - case_id: a
    ticker: AAPL
    query: x
    as_of: 2024-01-01
""",
    )
    universe = load_universe(path)
    assert universe.horizon_days == 30
    assert universe.signal_threshold == pytest.approx(0.4)
    assert universe.benchmark_ticker == "QQQ"
    assert universe.top_k == 12


def test_missing_required_field_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
cases:
  - ticker: NVDA
    query: x
    as_of: 2024-01-01
""",
    )
    with pytest.raises(UniverseError, match="case_id"):
        load_universe(path)


def test_missing_as_of_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
cases:
  - case_id: a
    ticker: NVDA
    query: x
""",
    )
    with pytest.raises(UniverseError, match="as_of"):
        load_universe(path)


def test_bad_as_of_format_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
cases:
  - case_id: a
    ticker: NVDA
    query: x
    as_of: not-a-date
""",
    )
    with pytest.raises(UniverseError, match="invalid as_of"):
        load_universe(path)


def test_duplicate_case_ids_raise(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
cases:
  - {case_id: a, ticker: NVDA, query: x, as_of: 2024-01-01}
  - {case_id: a, ticker: AAPL, query: y, as_of: 2024-02-01}
""",
    )
    with pytest.raises(UniverseError, match="duplicate"):
        load_universe(path)


def test_negative_horizon_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
horizon_days: -5
cases:
  - {case_id: a, ticker: NVDA, query: x, as_of: 2024-01-01}
""",
    )
    with pytest.raises(UniverseError, match="horizon_days"):
        load_universe(path)


def test_empty_cases_list_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "cases: []\n")
    with pytest.raises(UniverseError, match="non-empty"):
        load_universe(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(UniverseError, match="not found"):
        load_universe(tmp_path / "nope.yaml")
