"""End-to-end smoke test for the agent graph.

Wires a real :class:`ResearchGraph` with the scripted LLM client and a
fake specialist so we can verify the LangGraph DAG routes correctly and
preserves state across nodes. The DB and HybridSearch are stubbed out;
SQL paths are covered by integration tests.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from alphamind.agents.critic import CriticNode
from alphamind.agents.graph import ResearchGraph
from alphamind.agents.router import RouterNode
from alphamind.agents.specialists.base import SpecialistBase
from alphamind.agents.state import (
    AgentState,
    Citation,
    Claim,
    GraphInput,
    SpecialistName,
    SpecialistReport,
)
from alphamind.agents.synthesizer import SynthesizerNode
from tests.unit.agents.conftest import ScriptedLLMClient

pytestmark = pytest.mark.asyncio


def _citation(ordinal: int, chunk_id: int) -> Citation:
    return Citation(
        ordinal=ordinal,
        chunk_id=chunk_id,
        filing_id=chunk_id * 10,
        ticker="NVDA",
        form="10-K",
        filing_date=date(2024, 1, 1),
        section="Item 7",
        text=f"source-{chunk_id} text",
    )


class _StubSpecialist(SpecialistBase):
    """Bypasses retrieval entirely; emits a canned report."""

    name: SpecialistName = "fundamentals"
    system_prompt = "stub"

    def __init__(self, report: SpecialistReport) -> None:
        self._report = report

    @property
    def domain(self) -> str:
        return "stub"

    async def __call__(
        self,
        _state: AgentState,
    ) -> dict[str, list[SpecialistReport]]:
        return {"specialist_reports": [self._report]}


async def test_end_to_end_pipeline_writes_every_state_field() -> None:
    canned_report = SpecialistReport(
        specialist="fundamentals",
        summary="data center is huge",
        claims=(Claim(text="DC grew 86% YoY", citations=(1,)),),
        citations=(_citation(ordinal=1, chunk_id=42),),
    )

    # FIFO: router, synthesizer, critic — fundamentals never calls LLM.
    llm = ScriptedLLMClient(
        [
            json.dumps({"specialists": ["fundamentals"], "rationale": "fundies"}),
            json.dumps(
                {
                    "answer": "Data center growth dominates [1].",
                    "bull": ["DC grew 86% [1]."],
                    "bear": ["Concentration risk [1]."],
                }
            ),
            json.dumps({"unsupported": [], "notes": "well-supported"}),
        ]
    )

    graph = ResearchGraph(
        router=RouterNode(llm=llm),
        specialists={"fundamentals": _StubSpecialist(canned_report)},
        synthesizer=SynthesizerNode(llm=llm),
        critic=CriticNode(llm=llm),
    )

    state = await graph.invoke(
        GraphInput(query="NVDA bull/bear", as_of=date(2024, 12, 31), top_k=5)
    )

    decision = state["router_decision"]
    assert decision.specialists == ("fundamentals",)

    reports = state["specialist_reports"]
    assert len(reports) == 1
    assert reports[0].claims[0].text == "DC grew 86% YoY"

    thesis = state["thesis"]
    assert "Data center" in thesis.answer
    assert thesis.bull == ("DC grew 86% [1].",)
    assert thesis.bear == ("Concentration risk [1].",)
    assert [c.chunk_id for c in thesis.citations] == [42]

    critic = state["critic_report"]
    assert critic.ok
    assert critic.notes == "well-supported"


async def test_router_skip_short_circuits_to_synthesizer() -> None:
    """If the router picks specialists the graph hasn't registered, we
    should fall straight to the synthesizer rather than hang."""

    llm = ScriptedLLMClient(
        [
            # Router picks a specialist that ISN'T registered with the graph.
            json.dumps({"specialists": ["technical"], "rationale": "price action"}),
            # Synthesizer is still called; it produces the empty-thesis
            # fallback so the LLM script doesn't actually need to run.
            # ... but make sure we still hand the LLM a valid response in
            # case it does.
            json.dumps({"answer": "", "bull": [], "bear": []}),
            json.dumps({"unsupported": [], "notes": "nothing to critique"}),
        ]
    )

    graph = ResearchGraph(
        router=RouterNode(llm=llm),
        # Only fundamentals is registered.
        specialists={
            "fundamentals": _StubSpecialist(
                SpecialistReport(
                    specialist="fundamentals",
                    summary="s",
                    claims=(),
                    citations=(),
                )
            )
        },
        synthesizer=SynthesizerNode(llm=llm),
        critic=CriticNode(llm=llm),
    )

    state = await graph.invoke(GraphInput(query="momentum?", as_of=date(2024, 1, 1), top_k=3))

    # No specialist report because no registered specialist was selected.
    assert state.get("specialist_reports", []) == []
    # Synthesizer still produced a thesis (empty / fallback).
    assert "thesis" in state


async def test_invoke_rejects_empty_query() -> None:
    graph = ResearchGraph(
        router=RouterNode(llm=ScriptedLLMClient([])),
        specialists={
            "fundamentals": _StubSpecialist(
                SpecialistReport(specialist="fundamentals", summary="", claims=(), citations=())
            )
        },
        synthesizer=SynthesizerNode(llm=ScriptedLLMClient([])),
        critic=CriticNode(llm=ScriptedLLMClient([])),
    )
    with pytest.raises(ValueError):
        await graph.invoke(GraphInput(query="   ", as_of=date(2024, 1, 1)))
