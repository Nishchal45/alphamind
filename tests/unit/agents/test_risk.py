"""Tests for the risk specialist node.

These mirror the fundamentals tests in shape — the two share their
implementation via :mod:`alphamind.agents.specialists._base`, but the
risk specialist is its own surface and its own contract, so it gets
its own coverage rather than relying on transitivity.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from alphamind.agents.specialists.risk import make_risk_node
from alphamind.agents.state import ResearchState, Source
from tests.unit.agents.conftest import ScriptedLLMClient

pytestmark = pytest.mark.asyncio


def _state() -> ResearchState:
    return ResearchState(
        query="What downside risks does NVDA's China exposure carry?",
        as_of=date(2024, 12, 31),
        top_k=3,
    )


async def test_emits_findings_tagged_with_risk(sample_sources: list[Source]) -> None:
    payload = json.dumps(
        {
            "findings": [
                {
                    "claim": "Export controls could materially impact China sales.",
                    "cited_chunk_ids": [103],
                },
                {
                    "claim": "Material concentration in a single geography (China, 17%).",
                    "cited_chunk_ids": [101, 103],
                },
            ]
        }
    )
    client = ScriptedLLMClient(responses=[payload])

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        return sample_sources

    node = make_risk_node(client, retrieve)
    update = await node(_state())

    findings = update["findings"]
    assert len(findings) == 2
    assert all(f.specialist == "risk" for f in findings)
    assert findings[0].cited_chunk_ids == (103,)
    assert findings[1].cited_chunk_ids == (101, 103)
    assert update["usage"][0].node == "risk"


async def test_drops_findings_with_invalid_cites(
    sample_sources: list[Source],
) -> None:
    payload = json.dumps(
        {
            "findings": [
                {"claim": "fabricated cite", "cited_chunk_ids": [9999]},
            ]
        }
    )
    client = ScriptedLLMClient(responses=[payload])

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        return sample_sources

    node = make_risk_node(client, retrieve)
    update = await node(_state())
    assert update["findings"] == []


async def test_no_sources_means_no_findings() -> None:
    client = ScriptedLLMClient(responses=[])

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        return []

    node = make_risk_node(client, retrieve)
    update = await node(_state())
    assert update["sources"] == []
    assert update["findings"] == []
    assert client.calls == []


async def test_parse_failure_emits_no_findings(sample_sources: list[Source]) -> None:
    client = ScriptedLLMClient(responses=["definitely not json"])

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        return sample_sources

    node = make_risk_node(client, retrieve)
    update = await node(_state())
    assert update["findings"] == []
    assert len(update["sources"]) == 3
