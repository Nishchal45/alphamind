"""Tests for the router node."""

from __future__ import annotations

import json
from datetime import date

import pytest

from alphamind.agents.router import RouterNode
from alphamind.agents.state import ALL_SPECIALISTS, AgentState
from tests.unit.agents.conftest import ScriptedLLMClient

pytestmark = pytest.mark.asyncio


def _state(query: str = "what is NVDA saying about China revenue?") -> AgentState:
    return AgentState(query=query, as_of=date(2024, 12, 31), top_k=5, specialist_reports=[])


async def test_router_returns_clean_decision() -> None:
    llm = ScriptedLLMClient(
        [
            json.dumps(
                {
                    "specialists": ["fundamentals", "risk"],
                    "rationale": "revenue concentration is a fundamentals + risk question",
                }
            )
        ]
    )

    update = await RouterNode(llm=llm)(_state())
    decision = update["router_decision"]
    assert decision.specialists == ("fundamentals", "risk")
    assert "concentration" in decision.rationale


async def test_router_dedupes_and_validates_specialist_names() -> None:
    llm = ScriptedLLMClient(
        [
            json.dumps(
                {
                    "specialists": ["fundamentals", "fundamentals", "wizard", "risk"],
                    "rationale": "ok",
                }
            )
        ]
    )

    update = await RouterNode(llm=llm)(_state())
    assert update["router_decision"].specialists == ("fundamentals", "risk")


async def test_router_handles_fenced_json() -> None:
    llm = ScriptedLLMClient(
        [
            "Here is the routing decision:\n```json\n"
            + json.dumps({"specialists": ["fundamentals"], "rationale": "easy"})
            + "\n```"
        ]
    )

    update = await RouterNode(llm=llm)(_state())
    assert update["router_decision"].specialists == ("fundamentals",)


async def test_router_falls_back_to_all_specialists_on_malformed_response() -> None:
    llm = ScriptedLLMClient(["this is not json at all"])
    update = await RouterNode(llm=llm)(_state())
    assert update["router_decision"].specialists == ALL_SPECIALISTS
    assert "fallback" in update["router_decision"].rationale


async def test_router_falls_back_when_empty_array_returned() -> None:
    llm = ScriptedLLMClient([json.dumps({"specialists": [], "rationale": "n/a"})])
    update = await RouterNode(llm=llm)(_state())
    assert update["router_decision"].specialists == ALL_SPECIALISTS


async def test_router_requires_non_empty_query() -> None:
    llm = ScriptedLLMClient([json.dumps({"specialists": ["fundamentals"], "rationale": ""})])
    with pytest.raises(ValueError):
        await RouterNode(llm=llm)(_state(query=""))
