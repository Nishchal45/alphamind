"""Tests for the synthesizer node."""

from __future__ import annotations

import json

import pytest

from alphamind.agents.state import Finding, ResearchState
from alphamind.agents.synthesizer import make_synthesizer_node
from tests.unit.agents.conftest import ScriptedLLMClient

pytestmark = pytest.mark.asyncio


def _state_with_findings() -> ResearchState:
    return ResearchState(
        query="NVDA bull / bear?",
        findings=[
            Finding(
                specialist="fundamentals",
                claim="China is 17% of revenue.",
                cited_chunk_ids=(101,),
            ),
            Finding(
                specialist="fundamentals",
                claim="Margins expanded.",
                cited_chunk_ids=(102,),
            ),
        ],
    )


async def test_builds_structured_thesis() -> None:
    payload = json.dumps(
        {
            "summary": "Two-sided picture: strong margins, geopolitical risk.",
            "bull_case": [
                {"claim": "Operating leverage from Data Center.", "cited_chunk_ids": [102]},
            ],
            "bear_case": [
                {"claim": "China concentration is material.", "cited_chunk_ids": [101]},
            ],
        }
    )
    client = ScriptedLLMClient(responses=[payload])
    node = make_synthesizer_node(client)

    update = await node(_state_with_findings())
    thesis = update["thesis"]
    assert "Two-sided" in thesis.summary
    assert thesis.bull_case[0].cited_chunk_ids == (102,)
    assert thesis.bear_case[0].cited_chunk_ids == (101,)


async def test_drops_claims_with_invalid_cites() -> None:
    payload = json.dumps(
        {
            "summary": "ok",
            "bull_case": [
                {"claim": "fabricated cite", "cited_chunk_ids": [9999]},
                {"claim": "good cite", "cited_chunk_ids": [102]},
            ],
            "bear_case": [],
        }
    )
    client = ScriptedLLMClient(responses=[payload])
    node = make_synthesizer_node(client)

    thesis = (await node(_state_with_findings()))["thesis"]
    assert len(thesis.bull_case) == 1
    assert thesis.bull_case[0].claim == "good cite"


async def test_no_findings_means_empty_thesis() -> None:
    client = ScriptedLLMClient(responses=[])
    node = make_synthesizer_node(client)

    update = await node(ResearchState(query="anything", findings=[]))
    assert update["thesis"].bull_case == ()
    assert update["thesis"].bear_case == ()
    assert client.calls == []


async def test_parse_failure_emits_empty_thesis() -> None:
    client = ScriptedLLMClient(responses=["not json"])
    node = make_synthesizer_node(client)

    update = await node(_state_with_findings())
    assert update["thesis"].bull_case == ()
    assert update["thesis"].bear_case == ()
    assert "failed" in update["thesis"].summary.lower()
