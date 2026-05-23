"""Tests for the router node."""

from __future__ import annotations

import json
from datetime import date

import pytest

from alphamind.agents.router import make_router_node
from alphamind.agents.state import ResearchState
from tests.unit.agents.conftest import ScriptedLLMClient

pytestmark = pytest.mark.asyncio


def _state() -> ResearchState:
    return ResearchState(
        query="What is NVDA's China exposure?",
        as_of=date(2024, 12, 31),
        top_k=4,
    )


async def test_parses_intent_from_clean_json() -> None:
    payload = json.dumps(
        {
            "primary": "fundamentals",
            "specialists": ["fundamentals", "risk"],
            "rationale": "China exposure is in the fundamentals + risk sections.",
        }
    )
    client = ScriptedLLMClient(responses=[payload])
    node = make_router_node(client)

    update = await node(_state())

    intent = update["intent"]
    assert intent.primary == "fundamentals"
    assert intent.specialists == ("fundamentals", "risk")
    assert "China exposure" in intent.rationale
    assert update["usage"][0].node == "router"


async def test_inserts_primary_if_missing_from_specialists() -> None:
    payload = json.dumps({"primary": "risk", "specialists": ["fundamentals"], "rationale": "ok"})
    client = ScriptedLLMClient(responses=[payload])
    node = make_router_node(client)

    update = await node(_state())

    assert update["intent"].specialists[0] == "risk"
    assert "fundamentals" in update["intent"].specialists


async def test_falls_back_on_unparseable_response() -> None:
    client = ScriptedLLMClient(responses=["maybe fundamentals, maybe not"])
    node = make_router_node(client)

    update = await node(_state())

    intent = update["intent"]
    assert intent.primary == "fundamentals"
    assert intent.specialists == ("fundamentals",)
    assert intent.rationale == "parse_failure"


async def test_rejects_unknown_specialist() -> None:
    payload = json.dumps(
        {
            "primary": "astrology",
            "specialists": ["astrology", "fundamentals"],
            "rationale": "—",
        }
    )
    client = ScriptedLLMClient(responses=[payload])
    node = make_router_node(client)

    update = await node(_state())

    intent = update["intent"]
    assert intent.primary == "fundamentals"
    assert "astrology" not in intent.specialists
