"""Tests for the critic node."""

from __future__ import annotations

import json
from datetime import date

import pytest

from alphamind.agents.critic import CriticNode
from alphamind.agents.state import AgentState, Citation, Thesis
from tests.unit.agents.conftest import ScriptedLLMClient

pytestmark = pytest.mark.asyncio


def _citation(ordinal: int) -> Citation:
    return Citation(
        ordinal=ordinal,
        chunk_id=ordinal,
        filing_id=ordinal,
        ticker="NVDA",
        form="10-K",
        filing_date=date(2024, 1, 1),
        section=None,
        text=f"source {ordinal} body",
    )


def _state(thesis: Thesis | None) -> AgentState:
    state = AgentState(
        query="bull and bear case for NVDA",
        as_of=date(2024, 12, 31),
        top_k=5,
        specialist_reports=[],
    )
    if thesis is not None:
        state["thesis"] = thesis
    return state


async def test_critic_returns_unsupported_claims() -> None:
    thesis = Thesis(
        answer="Revenue tripled [1]. The product is also magical.",
        bull=("Magical product.",),
        bear=(),
        citations=(_citation(1),),
    )

    llm = ScriptedLLMClient(
        [
            json.dumps(
                {
                    "unsupported": [
                        {
                            "claim": "Magical product.",
                            "reason": "Source 1 says nothing about magic.",
                        }
                    ],
                    "notes": "One unsupported bullet flagged.",
                }
            )
        ]
    )

    update = await CriticNode(llm=llm)(_state(thesis))
    report = update["critic_report"]
    assert not report.ok
    assert len(report.unsupported) == 1
    assert "Magical" in report.unsupported[0].claim
    assert "magic" in report.unsupported[0].reason


async def test_critic_returns_clean_report_when_all_supported() -> None:
    thesis = Thesis(
        answer="Revenue tripled [1].",
        bull=("Revenue tripled [1].",),
        bear=(),
        citations=(_citation(1),),
    )
    llm = ScriptedLLMClient([json.dumps({"unsupported": [], "notes": "looks clean"})])

    update = await CriticNode(llm=llm)(_state(thesis))
    report = update["critic_report"]
    assert report.ok
    assert report.notes == "looks clean"


async def test_critic_short_circuits_on_empty_thesis() -> None:
    empty = Thesis(answer="", bull=(), bear=(), citations=())
    llm = ScriptedLLMClient([])  # should never be called
    update = await CriticNode(llm=llm)(_state(empty))
    assert update["critic_report"].ok
    assert llm.calls == []


async def test_critic_fallback_does_not_block_thesis() -> None:
    thesis = Thesis(
        answer="claim [1]",
        bull=("claim [1]",),
        bear=(),
        citations=(_citation(1),),
    )
    llm = ScriptedLLMClient(["garbage non-json"])
    update = await CriticNode(llm=llm)(_state(thesis))
    report = update["critic_report"]
    assert report.unsupported == ()
    assert "critic failed" in report.notes
