"""Tests for the threshold gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphamind.eval.thresholds import (
    Threshold,
    ThresholdsError,
    evaluate_thresholds,
    load_thresholds,
)
from alphamind.eval.types import EvalReport, MetricSummary


def _report(**overrides: float | int) -> EvalReport:
    """Build an :class:`EvalReport` from metric-name → mean overrides.

    Every metric defaults to ``n=2`` so threshold checks fire; pass
    ``<metric>_n=0`` to suppress one (the threshold should then be
    skipped, not failed).
    """
    metrics = [
        "citation_coverage",
        "citation_validity",
        "hallucination_rate",
        "contradiction_rate",
        "topic_recall",
        "chunk_recall",
    ]
    aggregate = tuple(
        MetricSummary(
            name=name,
            mean=float(overrides.get(name, 1.0)),
            minimum=float(overrides.get(name, 1.0)),
            maximum=float(overrides.get(name, 1.0)),
            n=int(overrides.get(f"{name}_n", 2)),
        )
        for name in metrics
    )
    return EvalReport(per_case=(), aggregate=aggregate, n_cases=2, n_failed=0)


# ---------------------------------------------------------------------------
# Threshold dataclass invariants.
# ---------------------------------------------------------------------------


def test_threshold_rejects_unknown_metric() -> None:
    with pytest.raises(ThresholdsError, match="unknown metric"):
        Threshold(metric="not_a_metric", minimum=0.5)


def test_threshold_rejects_no_bound() -> None:
    with pytest.raises(ThresholdsError, match="exactly one"):
        Threshold(metric="citation_coverage")


def test_threshold_rejects_two_bounds() -> None:
    with pytest.raises(ThresholdsError, match="exactly one"):
        Threshold(metric="citation_coverage", minimum=0.1, maximum=0.9)


# ---------------------------------------------------------------------------
# Loader.
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, contents: str) -> Path:
    p = tmp_path / "thresholds.yaml"
    p.write_text(contents, encoding="utf-8")
    return p


def test_loader_parses_minimum_and_maximum(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """\
thresholds:
  - metric: citation_coverage
    minimum: 0.8
  - metric: hallucination_rate
    maximum: 0.2
""",
    )
    out = load_thresholds(p)
    assert len(out) == 2
    by_metric = {t.metric: t for t in out}
    assert by_metric["citation_coverage"].minimum == 0.8
    assert by_metric["hallucination_rate"].maximum == 0.2


def test_loader_rejects_duplicate_metric(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """\
thresholds:
  - metric: citation_coverage
    minimum: 0.8
  - metric: citation_coverage
    minimum: 0.5
""",
    )
    with pytest.raises(ThresholdsError, match="duplicate"):
        load_thresholds(p)


def test_loader_rejects_empty_list(tmp_path: Path) -> None:
    p = _write(tmp_path, "thresholds: []\n")
    with pytest.raises(ThresholdsError, match="non-empty"):
        load_thresholds(p)


def test_loader_rejects_non_numeric_bound(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """\
thresholds:
  - metric: citation_coverage
    minimum: "high"
""",
    )
    with pytest.raises(ThresholdsError, match="number"):
        load_thresholds(p)


def test_shipped_thresholds_file_parses() -> None:
    """The thresholds file checked into the repo must always parse."""
    thresholds = load_thresholds(Path("evals/thresholds.yaml"))
    assert thresholds
    metrics = {t.metric for t in thresholds}
    # All six harness metrics are gated by the shipped defaults.
    assert metrics == {
        "citation_coverage",
        "citation_validity",
        "hallucination_rate",
        "contradiction_rate",
        "topic_recall",
        "chunk_recall",
    }


# ---------------------------------------------------------------------------
# evaluate_thresholds.
# ---------------------------------------------------------------------------


def test_no_violations_when_everything_passes() -> None:
    report = _report(
        citation_coverage=1.0,
        citation_validity=1.0,
        hallucination_rate=0.0,
        contradiction_rate=0.0,
        topic_recall=1.0,
        chunk_recall=1.0,
    )
    thresholds = [
        Threshold("citation_coverage", minimum=0.8),
        Threshold("hallucination_rate", maximum=0.2),
    ]
    assert evaluate_thresholds(report, thresholds) == []


def test_minimum_violation_records_side_minimum() -> None:
    report = _report(citation_coverage=0.4)
    [violation] = evaluate_thresholds(
        report,
        [Threshold("citation_coverage", minimum=0.8)],
    )
    assert violation.metric == "citation_coverage"
    assert violation.side == "minimum"
    assert violation.bound == 0.8
    assert violation.actual == pytest.approx(0.4)
    assert "below" in violation.message()


def test_maximum_violation_records_side_maximum() -> None:
    report = _report(hallucination_rate=0.5)
    [violation] = evaluate_thresholds(
        report,
        [Threshold("hallucination_rate", maximum=0.2)],
    )
    assert violation.side == "maximum"
    assert violation.bound == 0.2
    assert "above" in violation.message()


def test_unmeasured_metric_is_skipped_not_failed() -> None:
    # topic_recall n=0 means no case supplied expected_topics. The
    # threshold on that metric should be skipped silently.
    report = _report(topic_recall=0.0, topic_recall_n=0)
    assert evaluate_thresholds(report, [Threshold("topic_recall", minimum=0.5)]) == []
