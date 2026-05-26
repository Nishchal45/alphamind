"""Markdown + JSON + chart writers for the backtest report.

The markdown writer is the human-facing artefact. Per ADR 0010 the
methodological caveats open the file — they're the first thing
anyone reads, not buried at the end. The JSON sidecar is the
machine-readable companion. The chart is the equity curve.

Matplotlib is imported lazily inside :func:`write_chart` so the
package-level imports stay cheap (the rest of the harness doesn't
need matplotlib in the import graph).
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from alphamind.backtest.metrics import equity_curve
from alphamind.backtest.types import BacktestReport, CaseResult, Summary

CAVEAT_HEADER = """\
> **Methodological caveats.** This harness measures whether the agent
> system's expressed view is correlated with subsequent price action
> over a small sample of hand-picked cases. It is a sanity check —
> *not* a measurement of tradable alpha. No transaction costs, no
> slippage, no risk-adjusted sizing, no survivorship adjustment, no
> out-of-sample split. The aggregate metrics are descriptive on this
> sample size. Nothing in this report is financial advice.
"""


def write_markdown(report: BacktestReport, path: Path, *, chart_path: Path | None = None) -> None:
    """Write the human-facing report markdown to ``path``."""

    summary = report.summary
    lines: list[str] = ["# Backtest report\n", CAVEAT_HEADER, ""]

    lines.append("## Configuration\n")
    lines.append(f"- horizon_days: `{report.universe.horizon_days}`")
    lines.append(f"- signal_threshold: `{report.universe.signal_threshold}`")
    lines.append(f"- benchmark: `{report.universe.benchmark_ticker}`")
    lines.append(f"- top_k per specialist: `{report.universe.top_k}`")
    lines.append("")

    lines.append("## Aggregate metrics\n")
    if summary is None:
        lines.append("_no summary; the run produced no results._\n")
    else:
        lines.extend(_summary_rows(summary))
    lines.append("")

    if chart_path is not None:
        if chart_path.is_relative_to(path.parent):
            rel: Path = chart_path.relative_to(path.parent)
        else:
            rel = chart_path
        lines.append(f"![Equity curve]({rel})\n")

    lines.append("## Per-case results\n")
    lines.extend(_case_table(report.results))
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(report: BacktestReport, path: Path) -> None:
    """Write the JSON sidecar — exact dataclass dump, JSON-safe encoded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "universe": _to_jsonable(asdict(report.universe)),
        "results": [_to_jsonable(asdict(r)) for r in report.results],
        "summary": _to_jsonable(asdict(report.summary)) if report.summary else None,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_chart(report: BacktestReport, path: Path) -> None:
    """Render the equity curve to ``path`` as PNG.

    Skips if there's no active case (no curve to plot). Matplotlib
    is imported lazily so the rest of the harness doesn't pay the
    import cost.
    """
    curve = equity_curve(report.results)
    if not curve:
        return
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, no display required
    import matplotlib.pyplot as plt  # noqa: PLC0415

    case_ids = [pt[0] for pt in curve]
    equities = [pt[1] for pt in curve]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(equities)), equities, marker="o", linewidth=1.5)
    ax.axhline(y=1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(case_ids)))
    ax.set_xticklabels(case_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Equity (notional 1.0)")
    ax.set_title("Backtest equity curve (equal-weight, position-aware)")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _summary_rows(summary: Summary) -> list[str]:
    def _fmt(value: float | None, *, pct: bool = False, ratio: bool = False) -> str:
        if value is None:
            return "_n/a_"
        if pct:
            return f"{value * 100:+.2f}%"
        if ratio:
            return f"{value:.3f}"
        return f"{value}"

    return [
        f"- **n_cases**: {summary.n_cases}",
        f"- **n_active**: {summary.n_active} (cases that took a position)",
        f"- **n_errors**: {summary.n_errors}",
        f"- **hit_rate**: {_fmt(summary.hit_rate, ratio=True)}",
        f"- **mean_alpha** vs benchmark: {_fmt(summary.mean_alpha, pct=True)}",
        f"- **cagr** (active span, descriptive): {_fmt(summary.cagr, pct=True)}",
        f"- **sharpe** (active span, descriptive): {_fmt(summary.sharpe, ratio=True)}",
        f"- **max_drawdown**: {_fmt(summary.max_drawdown, pct=True)}",
    ]


def _case_table(results: tuple[CaseResult, ...] | list[CaseResult]) -> list[str]:
    header = (
        "| case_id | ticker | as_of | position | signal | ret | spy_ret | alpha | correct | error |"
    )
    divider = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    rows = [header, divider]
    for r in results:
        rows.append(
            "| "
            + " | ".join(
                [
                    r.case_id,
                    r.ticker,
                    r.as_of.isoformat(),
                    r.signal.position,
                    f"{r.signal.score:+.3f}",
                    _pct(r.position_return),
                    _pct(r.spy_return),
                    _pct(r.alpha),
                    _bool(r.correct),
                    r.error or "",
                ]
            )
            + " |"
        )
    return rows


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:+.2f}%"


def _bool(value: bool | None) -> str:
    if value is None:
        return "—"
    return "✓" if value else "✗"


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(x) for x in obj]
    return obj


__all__ = ["CAVEAT_HEADER", "write_chart", "write_json", "write_markdown"]
