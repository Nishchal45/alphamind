"""Tests for the sentiment specialist node.

Mirrors the fundamentals / risk tests in shape — the three share their
implementation via :mod:`alphamind.agents.specialists._base`, but each
specialist is its own surface and its own contract, so they each get
their own coverage rather than relying on transitivity.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from alphamind.agents.specialists.sentiment import make_sentiment_node
from alphamind.agents.state import ResearchState, Source
from tests.unit.agents.conftest import ScriptedLLMClient

pytestmark = pytest.mark.asyncio


def _state() -> ResearchState:
    return ResearchState(
        query="Has management's tone on China shifted in the latest filings?",
        as_of=date(2024, 12, 31),
        top_k=3,
    )


async def test_emits_findings_tagged_with_sentiment(
    sample_sources: list[Source],
) -> None:
    payload = json.dumps(
        {
            "findings": [
                {
                    "claim": (
                        "Management uses uncharacteristically hedged language "
                        "on China exposure ('could materially impact')."
                    ),
                    "cited_chunk_ids": [103],
                },
                {
                    "claim": "Forward-looking framing dominates the China commentary.",
                    "cited_chunk_ids": [101, 103],
                },
            ]
        }
    )
    client = ScriptedLLMClient(responses=[payload])

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        return sample_sources

    node = make_sentiment_node(client, retrieve)
    update = await node(_state())

    findings = update["findings"]
    assert len(findings) == 2
    assert all(f.specialist == "sentiment" for f in findings)
    assert findings[0].cited_chunk_ids == (103,)
    assert findings[1].cited_chunk_ids == (101, 103)
    assert update["usage"][0].node == "sentiment"


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

    node = make_sentiment_node(client, retrieve)
    update = await node(_state())
    assert update["findings"] == []


async def test_no_sources_means_no_findings() -> None:
    client = ScriptedLLMClient(responses=[])

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        return []

    node = make_sentiment_node(client, retrieve)
    update = await node(_state())
    assert update["sources"] == []
    assert update["findings"] == []
    assert client.calls == []


async def test_parse_failure_emits_no_findings(sample_sources: list[Source]) -> None:
    client = ScriptedLLMClient(responses=["definitely not json"])

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        return sample_sources

    node = make_sentiment_node(client, retrieve)
    update = await node(_state())
    assert update["findings"] == []
    assert len(update["sources"]) == 3
