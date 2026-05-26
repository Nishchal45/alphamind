"""YAML loader for backtest universes.

Mirrors the eval-harness loader (:mod:`alphamind.eval.loader`):
strict validation up front, clear exceptions, no silent type
coercion. The same ``as_of``-required discipline ADR 0005 enforces
elsewhere is enforced here at load time.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from alphamind.backtest.types import BacktestCase, BacktestUniverse

DEFAULT_HORIZON_DAYS = 90
DEFAULT_SIGNAL_THRESHOLD = 0.20
DEFAULT_BENCHMARK = "SPY"
DEFAULT_TOP_K = 8


class UniverseError(ValueError):
    """Raised when a YAML universe fails validation."""


def load_universe(path: Path) -> BacktestUniverse:
    """Read and validate a backtest universe YAML."""

    if not path.exists():
        raise UniverseError(f"universe file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise UniverseError(f"expected a top-level mapping in {path}, got {type(raw).__name__}")

    horizon_days = _coerce_int(raw, "horizon_days", DEFAULT_HORIZON_DAYS)
    if horizon_days <= 0:
        raise UniverseError(f"horizon_days must be positive, got {horizon_days}")

    threshold = _coerce_float(raw, "signal_threshold", DEFAULT_SIGNAL_THRESHOLD)
    if threshold < 0:
        raise UniverseError(f"signal_threshold must be non-negative, got {threshold}")

    benchmark = str(raw.get("benchmark_ticker", DEFAULT_BENCHMARK)).strip()
    if not benchmark:
        raise UniverseError("benchmark_ticker must be a non-empty string")

    top_k = _coerce_int(raw, "top_k", DEFAULT_TOP_K)
    if top_k <= 0:
        raise UniverseError(f"top_k must be positive, got {top_k}")

    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise UniverseError(f"'cases' must be a non-empty list in {path}")

    cases: list[BacktestCase] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(raw_cases):
        case = _parse_case(entry, index=i)
        if case.case_id in seen_ids:
            raise UniverseError(f"duplicate case_id: {case.case_id!r}")
        seen_ids.add(case.case_id)
        cases.append(case)

    return BacktestUniverse(
        cases=tuple(cases),
        horizon_days=horizon_days,
        signal_threshold=threshold,
        benchmark_ticker=benchmark,
        top_k=top_k,
    )


def _parse_case(entry: Any, *, index: int) -> BacktestCase:
    if not isinstance(entry, dict):
        raise UniverseError(f"case #{index}: expected mapping, got {type(entry).__name__}")
    missing = [k for k in ("case_id", "ticker", "query", "as_of") if k not in entry]
    if missing:
        raise UniverseError(f"case #{index}: missing required fields {missing}")

    case_id = str(entry["case_id"]).strip()
    ticker = str(entry["ticker"]).strip().upper()
    query = str(entry["query"]).strip()
    raw_as_of = entry["as_of"]

    if not case_id:
        raise UniverseError(f"case #{index}: case_id must be non-empty")
    if not ticker:
        raise UniverseError(f"case #{index}: ticker must be non-empty")
    if not query:
        raise UniverseError(f"case #{index}: query must be non-empty")

    as_of = _coerce_date(raw_as_of, where=f"case #{index} ({case_id})")

    return BacktestCase(case_id=case_id, ticker=ticker, query=query, as_of=as_of)


def _coerce_int(raw: dict[str, Any], key: str, default: int) -> int:
    if key not in raw:
        return default
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise UniverseError(f"{key} must be an integer, got {value!r}")
    return int(value)


def _coerce_float(raw: dict[str, Any], key: str, default: float) -> float:
    if key not in raw:
        return default
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise UniverseError(f"{key} must be a number, got {value!r}")
    return float(value)


def _coerce_date(raw: Any, *, where: str) -> date:
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, str):
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError as exc:
            raise UniverseError(f"{where}: invalid as_of {raw!r} (expected YYYY-MM-DD)") from exc
    raise UniverseError(
        f"{where}: as_of must be a date or YYYY-MM-DD string, got {type(raw).__name__}"
    )


__all__ = [
    "DEFAULT_BENCHMARK",
    "DEFAULT_HORIZON_DAYS",
    "DEFAULT_SIGNAL_THRESHOLD",
    "DEFAULT_TOP_K",
    "UniverseError",
    "load_universe",
]
