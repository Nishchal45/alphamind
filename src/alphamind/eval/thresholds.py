"""Threshold gating for the eval harness.

ADR 0008 ships the harness as a thermometer — measure six metrics,
write a report, no pass/fail. This module is the follow-up: a
thresholds YAML, loader, and a checker that turns an :class:`EvalReport`
into a list of :class:`ThresholdViolation`. The CLI plugs that list
into its exit code.

Each threshold pins either a ``min`` (metric must be ``>= min``) or
a ``max`` (metric must be ``<= max``), never both — every metric in
the harness is one-sided: coverage / recall / validity want high
numbers, hallucination / contradiction rates want low. A threshold
that tries to be two-sided is almost certainly a bug in the
thresholds file; the loader rejects it.

The check is over the *aggregate* (mean across cases), not per-case.
Per-case spikes are interesting but noisier; the aggregate is what we
gate CI on.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from alphamind.eval.types import EvalReport

# Mirrors the metric names produced by :func:`alphamind.eval.runner.run_eval`.
# Keep this in sync with the aggregate tuple constructed there.
_KNOWN_METRICS: frozenset[str] = frozenset(
    {
        "citation_coverage",
        "citation_validity",
        "hallucination_rate",
        "contradiction_rate",
        "topic_recall",
        "chunk_recall",
    }
)


class ThresholdsError(ValueError):
    """Raised when a thresholds file fails to parse or validate."""


@dataclass(frozen=True, slots=True)
class Threshold:
    """One bound on one metric.

    Exactly one of ``minimum`` / ``maximum`` must be set. ``minimum``
    means "the aggregate mean must be greater than or equal to this";
    ``maximum`` is the other direction. Two-sided gates aren't
    supported — every metric here is one-sided by intent.
    """

    metric: str
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if self.metric not in _KNOWN_METRICS:
            raise ThresholdsError(f"unknown metric {self.metric!r}")
        if (self.minimum is None) == (self.maximum is None):
            raise ThresholdsError(f"{self.metric}: exactly one of minimum / maximum must be set")


@dataclass(frozen=True, slots=True)
class ThresholdViolation:
    """One failed gate.

    ``side`` records which bound failed so the CLI's printed message
    can be specific without re-reading the threshold.
    """

    metric: str
    bound: float
    actual: float
    side: Literal["minimum", "maximum"]

    def message(self) -> str:
        direction = "below" if self.side == "minimum" else "above"
        return (
            f"{self.metric}: {self.actual:.3f} {direction} {self.side} threshold {self.bound:.3f}"
        )


def _coerce_threshold(raw: dict[str, Any]) -> Threshold:
    if "metric" not in raw:
        raise ThresholdsError("threshold entry missing 'metric'")
    metric = str(raw["metric"])
    minimum = raw.get("minimum")
    maximum = raw.get("maximum")

    def _num_or_none(value: Any, field: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ThresholdsError(f"{metric}: {field} must be a number, got {value!r}")
        return float(value)

    return Threshold(
        metric=metric,
        minimum=_num_or_none(minimum, "minimum"),
        maximum=_num_or_none(maximum, "maximum"),
    )


def load_thresholds(path: Path) -> list[Threshold]:
    """Parse a YAML thresholds file into a list of :class:`Threshold`."""
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ThresholdsError(f"{path}: top-level value must be a mapping")
    raw = data.get("thresholds")
    if not isinstance(raw, list) or not raw:
        raise ThresholdsError(f"{path}: 'thresholds' must be a non-empty list")

    seen: set[str] = set()
    out: list[Threshold] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ThresholdsError(f"thresholds[{i}]: must be a mapping")
        threshold = _coerce_threshold(item)
        if threshold.metric in seen:
            raise ThresholdsError(f"thresholds[{i}]: duplicate threshold for {threshold.metric!r}")
        seen.add(threshold.metric)
        out.append(threshold)
    return out


def evaluate_thresholds(
    report: EvalReport,
    thresholds: Iterable[Threshold],
) -> list[ThresholdViolation]:
    """Score a report against a set of thresholds.

    A threshold whose metric has ``n == 0`` in the report (no case
    contributed) is skipped, not failed — gating on a metric we
    didn't measure would be louder than useful. The CLI surfaces the
    skipped metric in the summary so it's visible.
    """
    aggregate_by_name = {m.name: m for m in report.aggregate}
    violations: list[ThresholdViolation] = []

    for threshold in thresholds:
        summary = aggregate_by_name.get(threshold.metric)
        if summary is None or summary.n == 0:
            continue
        if threshold.minimum is not None and summary.mean < threshold.minimum:
            violations.append(
                ThresholdViolation(
                    metric=threshold.metric,
                    bound=threshold.minimum,
                    actual=summary.mean,
                    side="minimum",
                )
            )
        if threshold.maximum is not None and summary.mean > threshold.maximum:
            violations.append(
                ThresholdViolation(
                    metric=threshold.metric,
                    bound=threshold.maximum,
                    actual=summary.mean,
                    side="maximum",
                )
            )

    return violations


__all__ = [
    "Threshold",
    "ThresholdViolation",
    "ThresholdsError",
    "evaluate_thresholds",
    "load_thresholds",
]
