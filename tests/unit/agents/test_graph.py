"""End-to-end test for the compiled LangGraph DAG."""

from __future__ import annotations

import json
from datetime import date

import pytest

from alphamind.agents import build_research_graph
from alphamind.agents.state import Source
from tests.unit.agents.conftest import ScriptedLLMClient

pytestmark = pytest.mark.asyncio


async def test_full_dag_runs_router_fundamentals_synth_critic(
    sample_sources: list[Source],
) -> None:
    router_resp = json.dumps(
        {
            "primary": "fundamentals",
            "specialists": ["fundamentals"],
            "rationale": "Question is about revenue mix.",
        }
    )
    fundamentals_resp = json.dumps(
        {
            "findings": [
                {"claim": "China is 17% of revenue.", "cited_chunk_ids": [101]},
                {"claim": "Gross margin reached 73%.", "cited_chunk_ids": [102]},
            ]
        }
    )
    synthesizer_resp = json.dumps(
        {
            "summary": "Margins are strong, but China concentration is a risk.",
            "bull_case": [
                {"claim": "Gross margin expanded.", "cited_chunk_ids": [102]},
            ],
            "bear_case": [
                {"claim": "China revenue concentration.", "cited_chunk_ids": [101]},
            ],
        }
    )
    critic_resp = json.dumps({"issues": []})

    client = ScriptedLLMClient(
        responses=[router_resp, fundamentals_resp, synthesizer_resp, critic_resp],
    )

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        return sample_sources

    graph = build_research_graph(llm=client, retrieve=retrieve)
    result = await graph.ainvoke(
        {
            "query": "What is NVDA's China revenue exposure?",
            "as_of": date(2024, 12, 31),
            "top_k": 3,
        }
    )

    assert result["intent"].primary == "fundamentals"
    assert {s.chunk_id for s in result["sources"]} == {101, 102, 103}

    findings = list(result["findings"])
    assert len(findings) == 2

    thesis = result["thesis"]
    assert "China" in thesis.summary or "margin" in thesis.summary.lower()
    assert thesis.bull_case[0].cited_chunk_ids == (102,)
    assert thesis.bear_case[0].cited_chunk_ids == (101,)

    assert result["critique"].issues == ()
    assert result["critique"].parse_error is None

    # All four nodes contributed to usage.
    nodes_in_usage = {u.node for u in result["usage"]}
    assert nodes_in_usage == {"router", "fundamentals", "synthesizer", "critic"}


async def test_dag_survives_dud_router_response(sample_sources: list[Source]) -> None:
    """A dud router response should still drive the DAG to a thesis via the fallback path."""

    fundamentals_resp = json.dumps(
        {
            "findings": [
                {"claim": "Margins reached 73%.", "cited_chunk_ids": [102]},
            ]
        }
    )
    synthesizer_resp = json.dumps(
        {
            "summary": "Strong margins.",
            "bull_case": [{"claim": "Margins.", "cited_chunk_ids": [102]}],
            "bear_case": [],
        }
    )
    critic_resp = json.dumps({"issues": []})

    client = ScriptedLLMClient(
        responses=["sure thing!", fundamentals_resp, synthesizer_resp, critic_resp],
    )

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        return sample_sources

    graph = build_research_graph(llm=client, retrieve=retrieve)
    result = await graph.ainvoke(
        {
            "query": "Anything",
            "as_of": date(2024, 12, 31),
            "top_k": 3,
        }
    )

    assert result["intent"].rationale == "parse_failure"
    assert len(result["thesis"].bull_case) == 1
