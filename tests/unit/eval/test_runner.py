"""Test the eval runner end-to-end against a stub-wired research graph."""

from __future__ import annotations

import json
from datetime import date

import pytest

from alphamind.agents import build_research_graph
from alphamind.agents.state import Source
from alphamind.eval.runner import run_eval
from alphamind.eval.types import EvalCase
from tests.unit.agents.conftest import SystemKeyedLLMClient

pytestmark = pytest.mark.asyncio


def _case(case_id: str, **kwargs: object) -> EvalCase:
    return EvalCase(
        id=case_id,
        query=str(kwargs.pop("query", "what is X?")),
        as_of=kwargs.pop("as_of", date(2024, 12, 31)),  # type: ignore[arg-type]
        expected_topics=tuple(kwargs.pop("expected_topics", ())),  # type: ignore[arg-type]
        required_chunk_ids=tuple(kwargs.pop("required_chunk_ids", ())),  # type: ignore[arg-type]
    )


async def test_runner_aggregates_metrics_across_cases() -> None:
    router_resp = json.dumps(
        {
            "primary": "fundamentals",
            "specialists": ["fundamentals", "risk"],
            "rationale": "two-sided",
        }
    )
    fundamentals_resp = json.dumps(
        {"findings": [{"claim": "Gross margin 73%.", "cited_chunk_ids": [102]}]}
    )
    risk_resp = json.dumps(
        {"findings": [{"claim": "China export controls.", "cited_chunk_ids": [103]}]}
    )
    synthesizer_resp = json.dumps(
        {
            "summary": "Margins strong; China risk material.",
            "bull_case": [{"claim": "Gross margin expanded.", "cited_chunk_ids": [102]}],
            "bear_case": [{"claim": "China concentration.", "cited_chunk_ids": [103]}],
        }
    )
    critic_resp = json.dumps({"issues": []})

    client = SystemKeyedLLMClient(
        responses_by_marker={
            "routing layer": router_resp,
            "fundamentals specialist": fundamentals_resp,
            "risk specialist": risk_resp,
            "synthesizer": synthesizer_resp,
            "critic": critic_resp,
        }
    )

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        return [
            Source(
                chunk_id=102,
                filing_id=1,
                ticker="NVDA",
                form="10-K",
                filing_date=date(2024, 2, 1),
                section="Item 7",
                text="Gross margin expanded.",
                score=0.9,
            ),
            Source(
                chunk_id=103,
                filing_id=2,
                ticker="NVDA",
                form="10-Q",
                filing_date=date(2024, 8, 1),
                section="Item 1A",
                text="Export controls could impact China sales.",
                score=0.8,
            ),
        ]

    graph = build_research_graph(llm=client, retrieve=retrieve)

    cases = [
        _case("a", expected_topics=("China", "margin"), required_chunk_ids=(102, 103)),
        _case("b", expected_topics=("nowhere",)),  # 0 / 1
    ]
    report = await run_eval(graph, cases)

    assert report.n_cases == 2
    assert report.n_failed == 0

    # Healthy graph: every claim cites a real chunk, no critic issues.
    coverage = next(m for m in report.aggregate if m.name == "citation_coverage")
    assert coverage.mean == 1.0
    validity = next(m for m in report.aggregate if m.name == "citation_validity")
    assert validity.mean == 1.0

    halluc = next(m for m in report.aggregate if m.name == "hallucination_rate")
    assert halluc.mean == 0.0

    topic = next(m for m in report.aggregate if m.name == "topic_recall")
    # Case a: both 'China' and 'margin' present → 1.0; case b: 'nowhere' → 0.0.
    assert topic.mean == 0.5
    assert topic.n == 2

    chunk = next(m for m in report.aggregate if m.name == "chunk_recall")
    # Only case a supplied required_chunk_ids; both 102 and 103 are cited.
    assert chunk.mean == 1.0
    assert chunk.n == 1


async def test_runner_records_per_case_failure_without_aborting() -> None:
    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        raise RuntimeError("retrieval blew up")

    # A scripted response that produces a thesis isn't reached because retrieval
    # fails first; we just need the graph to be invokable.
    client = SystemKeyedLLMClient(responses_by_marker={}, default_response="{}")
    graph = build_research_graph(llm=client, retrieve=retrieve)

    cases = [_case("dead"), _case("also-dead")]
    report = await run_eval(graph, cases)

    assert report.n_cases == 2
    assert report.n_failed == 2
    assert all(r.failure is not None and "retrieval blew up" in r.failure for r in report.per_case)
