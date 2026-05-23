"""Typed records used by the eval harness.

The shape is deliberately narrow: a golden-set entry is one
:class:`EvalCase`, a graph run produces one :class:`CaseResult`, the
whole suite aggregates into one :class:`EvalReport`. Everything's
frozen + slots — same style as the agent and retrieval layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One question to grade the pipeline on.

    ``expected_topics`` and ``required_chunk_ids`` are both optional
    hints — a case can leave either empty and only the metrics that
    don't depend on them will be scored.

    ``id`` is the stable handle the report uses to identify a case;
    keep it short and grep-friendly.
    """

    id: str
    query: str
    as_of: date
    expected_topics: tuple[str, ...] = field(default_factory=tuple)
    required_chunk_ids: tuple[int, ...] = field(default_factory=tuple)
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Per-case scoring output.

    Holds the metric scalars and a small set of inspected fields lifted
    off the final :class:`ResearchState`. The full state isn't carried
    here — it's a fat object and the eval report ends up on disk; what
    we want persisted is the score plus enough context to triage a
    regression.
    """

    case_id: str
    citation_coverage: float
    citation_validity: float
    hallucination_rate: float
    contradiction_rate: float
    topic_recall: float | None
    chunk_recall: float | None
    n_bull_claims: int
    n_bear_claims: int
    n_critic_issues: int
    n_sources: int
    total_input_tokens: int
    total_output_tokens: int
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Aggregate of one metric across all cases."""

    name: str
    mean: float
    minimum: float
    maximum: float
    n: int  # cases that contributed (skipped Nones don't count)


@dataclass(frozen=True, slots=True)
class EvalReport:
    """The whole suite's output.

    ``per_case`` preserves order so a diff against an earlier report
    lines up case-by-case. ``aggregate`` carries one
    :class:`MetricSummary` per metric.
    """

    per_case: tuple[CaseResult, ...]
    aggregate: tuple[MetricSummary, ...]
    n_cases: int
    n_failed: int
