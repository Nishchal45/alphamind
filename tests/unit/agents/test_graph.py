"""End-to-end test for the compiled LangGraph DAG."""

from __future__ import annotations

import json
from datetime import date

import pytest

from alphamind.agents import build_research_graph
from alphamind.agents.state import Source
from tests.unit.agents.conftest import ScriptedLLMClient, SystemKeyedLLMClient

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


async def test_dag_fans_out_to_both_specialists(sample_sources: list[Source]) -> None:
    """Router asking for both specialists should fire both in parallel and fan in at synth."""

    router_resp = json.dumps(
        {
            "primary": "risk",
            "specialists": ["fundamentals", "risk"],
            "rationale": "Question covers both revenue mix and regulatory exposure.",
        }
    )
    # Same response shape for both specialists; only the binding system
    # prompt determines which one this canned reply belongs to. Each
    # node tags its own findings with its specialist name.
    fundamentals_resp = json.dumps(
        {
            "findings": [
                {"claim": "Gross margin reached 73%.", "cited_chunk_ids": [102]},
            ]
        }
    )
    risk_resp = json.dumps(
        {
            "findings": [
                {
                    "claim": "Export controls could materially impact China sales.",
                    "cited_chunk_ids": [103],
                },
            ]
        }
    )
    synthesizer_resp = json.dumps(
        {
            "summary": "Margins strong; geopolitical risk is the dominant downside.",
            "bull_case": [
                {"claim": "Gross margin expanded.", "cited_chunk_ids": [102]},
            ],
            "bear_case": [
                {"claim": "Export-control risk to China revenue.", "cited_chunk_ids": [103]},
            ],
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
        return sample_sources

    graph = build_research_graph(llm=client, retrieve=retrieve)
    result = await graph.ainvoke(
        {
            "query": "Bull / bear on NVDA given China exposure?",
            "as_of": date(2024, 12, 31),
            "top_k": 3,
        }
    )

    specialists_seen = {f.specialist for f in result["findings"]}
    assert specialists_seen == {"fundamentals", "risk"}

    nodes_in_usage = {u.node for u in result["usage"]}
    assert nodes_in_usage == {"router", "fundamentals", "risk", "synthesizer", "critic"}

    # Both bull and bear sides land; each cites its expected specialist's chunk.
    assert result["thesis"].bull_case[0].cited_chunk_ids == (102,)
    assert result["thesis"].bear_case[0].cited_chunk_ids == (103,)


async def test_dag_fans_out_to_all_four_specialists(
    sample_sources: list[Source],
) -> None:
    """Router asking for all four specialists should fire all four and fan in at synth."""

    router_resp = json.dumps(
        {
            "primary": "fundamentals",
            "specialists": ["fundamentals", "risk", "sentiment", "technical"],
            "rationale": "Broad question covering financials, risks, tone, and trends.",
        }
    )
    fundamentals_resp = json.dumps(
        {
            "findings": [
                {"claim": "Gross margin reached 73%.", "cited_chunk_ids": [102]},
            ]
        }
    )
    risk_resp = json.dumps(
        {
            "findings": [
                {
                    "claim": "Export controls could materially impact China sales.",
                    "cited_chunk_ids": [103],
                },
            ]
        }
    )
    sentiment_resp = json.dumps(
        {
            "findings": [
                {
                    "claim": "Hedged language ('could materially impact') around China exposure.",
                    "cited_chunk_ids": [103],
                },
            ]
        }
    )
    technical_resp = json.dumps(
        {
            "findings": [
                {
                    "claim": "Gross margin expanding as Data Center mix grows.",
                    "cited_chunk_ids": [102],
                },
            ]
        }
    )
    synthesizer_resp = json.dumps(
        {
            "summary": "Margins expanding; geopolitical risk is the dominant downside.",
            "bull_case": [
                {"claim": "Gross margin expanded.", "cited_chunk_ids": [102]},
            ],
            "bear_case": [
                {"claim": "Export-control risk to China revenue.", "cited_chunk_ids": [103]},
            ],
        }
    )
    critic_resp = json.dumps({"issues": []})

    client = SystemKeyedLLMClient(
        responses_by_marker={
            "routing layer": router_resp,
            "fundamentals specialist": fundamentals_resp,
            "risk specialist": risk_resp,
            "sentiment specialist": sentiment_resp,
            "technical specialist": technical_resp,
            "synthesizer": synthesizer_resp,
            "critic": critic_resp,
        }
    )

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        return sample_sources

    graph = build_research_graph(llm=client, retrieve=retrieve)
    result = await graph.ainvoke(
        {
            "query": "Full bull / bear on NVDA covering financials, risk, tone, and trends?",
            "as_of": date(2024, 12, 31),
            "top_k": 3,
        }
    )

    specialists_seen = {f.specialist for f in result["findings"]}
    assert specialists_seen == {"fundamentals", "risk", "sentiment", "technical"}

    nodes_in_usage = {u.node for u in result["usage"]}
    assert nodes_in_usage == {
        "router",
        "fundamentals",
        "risk",
        "sentiment",
        "technical",
        "synthesizer",
        "critic",
    }

    # Critic accepts a thesis cleanly when all citations are well-formed.
    assert result["critique"].issues == ()


async def test_router_omitting_fundamentals_still_runs_it(
    sample_sources: list[Source],
) -> None:
    """The graph should force fundamentals into the fan-out as a safety net."""

    router_resp = json.dumps(
        {
            "primary": "risk",
            "specialists": ["risk"],  # router intentionally omits fundamentals
            "rationale": "Risk-only question.",
        }
    )
    fundamentals_resp = json.dumps(
        {"findings": [{"claim": "fund finding", "cited_chunk_ids": [101]}]}
    )
    risk_resp = json.dumps({"findings": [{"claim": "risk finding", "cited_chunk_ids": [103]}]})
    synthesizer_resp = json.dumps(
        {
            "summary": "ok",
            "bull_case": [{"claim": "b", "cited_chunk_ids": [101]}],
            "bear_case": [{"claim": "r", "cited_chunk_ids": [103]}],
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
        return sample_sources

    graph = build_research_graph(llm=client, retrieve=retrieve)
    result = await graph.ainvoke(
        {"query": "Risk only please", "as_of": date(2024, 12, 31), "top_k": 3}
    )

    specialists_seen = {f.specialist for f in result["findings"]}
    assert specialists_seen == {"fundamentals", "risk"}
