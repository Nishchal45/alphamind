"""Tests for the synthesizer node."""

from __future__ import annotations

import json
from datetime import date

import pytest

from alphamind.agents.state import (
    AgentState,
    Citation,
    Claim,
    SpecialistReport,
)
from alphamind.agents.synthesizer import SynthesizerNode, _merge_citations
from tests.unit.agents.conftest import ScriptedLLMClient

pytestmark = pytest.mark.asyncio


def _citation(*, ordinal: int, chunk_id: int, ticker: str = "NVDA") -> Citation:
    return Citation(
        ordinal=ordinal,
        chunk_id=chunk_id,
        filing_id=chunk_id * 10,
        ticker=ticker,
        form="10-K",
        filing_date=date(2024, 1, 1),
        section="Item 7",
        text=f"chunk-{chunk_id} body",
    )


def _report(name: str, citations: tuple[Citation, ...]) -> SpecialistReport:
    return SpecialistReport(
        specialist=name,  # type: ignore[arg-type]
        summary=f"{name} summary",
        claims=tuple(Claim(text=f"claim {i}", citations=(i + 1,)) for i in range(len(citations))),
        citations=citations,
    )


async def test_merge_citations_renumbers_and_dedupes() -> None:
    fund = _report(
        "fundamentals",
        (_citation(ordinal=1, chunk_id=101), _citation(ordinal=2, chunk_id=102)),
    )
    risk = _report(
        "risk",
        (_citation(ordinal=1, chunk_id=102), _citation(ordinal=2, chunk_id=103)),
    )

    merged = _merge_citations([fund, risk])
    assert [c.ordinal for c in merged] == [1, 2, 3]
    assert [c.chunk_id for c in merged] == [101, 102, 103]


async def test_synthesizer_produces_thesis() -> None:
    fund = _report(
        "fundamentals",
        (_citation(ordinal=1, chunk_id=1), _citation(ordinal=2, chunk_id=2)),
    )
    llm = ScriptedLLMClient(
        [
            json.dumps(
                {
                    "answer": "NVDA data-center is the dominant driver [1].",
                    "bull": ["Data-center revenue grew 86% [1].", "Margins expanded [2]."],
                    "bear": ["Concentration risk if data-center slows [2]."],
                }
            )
        ]
    )

    state = AgentState(
        query="bull and bear case",
        as_of=date(2024, 12, 31),
        top_k=5,
        specialist_reports=[fund],
    )
    update = await SynthesizerNode(llm=llm)(state)
    thesis = update["thesis"]

    assert thesis.answer.startswith("NVDA data-center")
    assert len(thesis.bull) == 2
    assert len(thesis.bear) == 1
    assert {c.chunk_id for c in thesis.citations} == {1, 2}


async def test_synthesizer_falls_back_on_malformed_response() -> None:
    fund = _report("fundamentals", (_citation(ordinal=1, chunk_id=1),))
    llm = ScriptedLLMClient(["this is not JSON"])
    state = AgentState(
        query="q",
        as_of=date(2024, 1, 1),
        top_k=5,
        specialist_reports=[fund],
    )
    update = await SynthesizerNode(llm=llm)(state)
    thesis = update["thesis"]

    # Fallback thesis carries the failure message in the answer and keeps
    # the citations intact so the critic still has the source pool.
    assert "synthesizer failed" in thesis.answer
    assert thesis.citations and thesis.citations[0].chunk_id == 1


async def test_synthesizer_short_circuits_without_reports() -> None:
    llm = ScriptedLLMClient([])  # should never be called
    state = AgentState(
        query="q",
        as_of=date(2024, 1, 1),
        top_k=5,
        specialist_reports=[],
    )
    update = await SynthesizerNode(llm=llm)(state)
    assert update["thesis"].answer.startswith("No specialist reports")
    assert llm.calls == []
