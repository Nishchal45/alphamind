"""Drive a compiled research graph across a golden set and score each run.

The runner is intentionally thin — it doesn't know how to *build*
the graph. Callers (`scripts/eval.py`, tests) hand it a compiled
graph plus the cases, and it walks them. That keeps the runner
testable against a stub-wired graph while the production CLI wires
the real one.

A per-case failure (graph raised, no thesis produced) is captured on
``CaseResult.failure`` and the case still appears in the report, with
zeroed metrics. We don't abort the suite — running the remaining cases
is more useful than crashing on the first regression.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Sequence
from typing import Any, cast

from langgraph.graph.state import CompiledStateGraph

from alphamind.agents.state import ResearchState
from alphamind.eval.metrics import (
    chunk_recall,
    citation_coverage,
    citation_validity,
    contradiction_rate,
    hallucination_rate,
    topic_recall,
)
from alphamind.eval.types import CaseResult, EvalCase, EvalReport, MetricSummary

logger = logging.getLogger(__name__)


def _summarise(
    name: str,
    values: Sequence[float | None],
) -> MetricSummary:
    real = [v for v in values if v is not None]
    if not real:
        return MetricSummary(name=name, mean=0.0, minimum=0.0, maximum=0.0, n=0)
    return MetricSummary(
        name=name,
        mean=statistics.fmean(real),
        minimum=min(real),
        maximum=max(real),
        n=len(real),
    )


def _zero_result(case: EvalCase, failure: str) -> CaseResult:
    return CaseResult(
        case_id=case.id,
        citation_coverage=0.0,
        citation_validity=0.0,
        hallucination_rate=0.0,
        contradiction_rate=0.0,
        topic_recall=None if not case.expected_topics else 0.0,
        chunk_recall=None if not case.required_chunk_ids else 0.0,
        n_bull_claims=0,
        n_bear_claims=0,
        n_critic_issues=0,
        n_sources=0,
        total_input_tokens=0,
        total_output_tokens=0,
        failure=failure,
    )


def _score(case: EvalCase, state: ResearchState) -> CaseResult:
    thesis = state.get("thesis")
    critique = state.get("critique")
    usage = state.get("usage", [])
    sources = state.get("sources", [])

    n_bull = len(thesis.bull_case) if thesis is not None else 0
    n_bear = len(thesis.bear_case) if thesis is not None else 0
    n_issues = len(critique.issues) if critique is not None else 0
    n_sources = len({s.chunk_id for s in sources})

    return CaseResult(
        case_id=case.id,
        citation_coverage=citation_coverage(state),
        citation_validity=citation_validity(state),
        hallucination_rate=hallucination_rate(state),
        contradiction_rate=contradiction_rate(state),
        topic_recall=topic_recall(state, case.expected_topics),
        chunk_recall=chunk_recall(state, case.required_chunk_ids),
        n_bull_claims=n_bull,
        n_bear_claims=n_bear,
        n_critic_issues=n_issues,
        n_sources=n_sources,
        total_input_tokens=sum(u.input_tokens for u in usage),
        total_output_tokens=sum(u.output_tokens for u in usage),
        failure=None,
    )


async def run_eval(
    graph: CompiledStateGraph[ResearchState, Any, Any, Any],
    cases: Sequence[EvalCase],
    *,
    top_k: int = 8,
) -> EvalReport:
    """Run ``cases`` through ``graph`` and assemble an :class:`EvalReport`."""
    results: list[CaseResult] = []

    for case in cases:
        logger.info("eval: running case %s", case.id)
        try:
            raw = await graph.ainvoke(
                {
                    "query": case.query,
                    "as_of": case.as_of,
                    "top_k": top_k,
                }
            )
            # ``ainvoke`` widens the static type; cast back to our
            # TypedDict so the metrics layer can read fields cleanly.
            state = cast("ResearchState", raw)
            results.append(_score(case, state))
        except Exception as exc:
            logger.exception("eval: case %s failed", case.id)
            results.append(_zero_result(case, str(exc)))

    aggregate = (
        _summarise("citation_coverage", [r.citation_coverage for r in results]),
        _summarise("citation_validity", [r.citation_validity for r in results]),
        _summarise("hallucination_rate", [r.hallucination_rate for r in results]),
        _summarise("contradiction_rate", [r.contradiction_rate for r in results]),
        _summarise("topic_recall", [r.topic_recall for r in results]),
        _summarise("chunk_recall", [r.chunk_recall for r in results]),
    )

    return EvalReport(
        per_case=tuple(results),
        aggregate=aggregate,
        n_cases=len(results),
        n_failed=sum(1 for r in results if r.failure is not None),
    )


__all__ = ["run_eval"]
