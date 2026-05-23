"""Evaluation harness — score the agent pipeline against a golden set.

The harness has three pieces:

- :mod:`alphamind.eval.types` — typed records: ``EvalCase`` (input),
  ``CaseResult`` (per-case metrics + the full :class:`ResearchState`),
  ``EvalReport`` (aggregate).
- :mod:`alphamind.eval.metrics` — pure functions over a finished
  :class:`ResearchState`: citation coverage, citation validity,
  hallucination rate, contradiction rate, topic recall, chunk recall.
- :mod:`alphamind.eval.runner` — runs a compiled graph across the
  cases and assembles the report.
- :mod:`alphamind.eval.loader` — YAML loader for the golden set.

Today the harness *measures* — it doesn't gate. A later slice may add
thresholds and a non-zero exit code when a metric drops below them.
For now the report is for a human to read.
"""

from __future__ import annotations

from alphamind.eval.loader import load_golden_set
from alphamind.eval.metrics import (
    chunk_recall,
    citation_coverage,
    citation_validity,
    contradiction_rate,
    hallucination_rate,
    topic_recall,
)
from alphamind.eval.runner import run_eval
from alphamind.eval.thresholds import (
    Threshold,
    ThresholdsError,
    ThresholdViolation,
    evaluate_thresholds,
    load_thresholds,
)
from alphamind.eval.types import CaseResult, EvalCase, EvalReport, MetricSummary

__all__ = [
    "CaseResult",
    "EvalCase",
    "EvalReport",
    "MetricSummary",
    "Threshold",
    "ThresholdViolation",
    "ThresholdsError",
    "chunk_recall",
    "citation_coverage",
    "citation_validity",
    "contradiction_rate",
    "evaluate_thresholds",
    "hallucination_rate",
    "load_golden_set",
    "load_thresholds",
    "run_eval",
    "topic_recall",
]
