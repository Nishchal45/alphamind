"""Tests for the fundamentals specialist node."""

from __future__ import annotations

import json
from datetime import date

import pytest

from alphamind.agents.specialists.fundamentals import make_fundamentals_node
from alphamind.agents.state import ResearchState, Source
from tests.unit.agents.conftest import ScriptedLLMClient

pytestmark = pytest.mark.asyncio


def _state() -> ResearchState:
    return ResearchState(
        query="What is NVDA's China exposure?",
        as_of=date(2024, 12, 31),
        top_k=3,
    )


async def test_emits_findings_with_valid_citations(
    sample_sources: list[Source],
) -> None:
    payload = json.dumps(
        {
            "findings": [
                {"claim": "China is 17% of revenue.", "cited_chunk_ids": [101]},
                {
                    "claim": "Margins expanded on Data Center mix.",
                    "cited_chunk_ids": [102, 103],
                },
            ]
        }
    )
    client = ScriptedLLMClient(responses=[payload])

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        return sample_sources

    node = make_fundamentals_node(client, retrieve)
    update = await node(_state())

    findings = update["findings"]
    assert len(findings) == 2
    assert findings[0].specialist == "fundamentals"
    assert findings[0].cited_chunk_ids == (101,)
    assert findings[1].cited_chunk_ids == (102, 103)

    # All retrieved chunks appear in the source pool.
    assert {s.chunk_id for s in update["sources"]} == {101, 102, 103}


async def test_drops_findings_with_invalid_chunk_ids(
    sample_sources: list[Source],
) -> None:
    payload = json.dumps(
        {
            "findings": [
                {"claim": "claim with invalid cite", "cited_chunk_ids": [9999]},
                {"claim": "claim with mixed cites", "cited_chunk_ids": [102, 9999]},
            ]
        }
    )
    client = ScriptedLLMClient(responses=[payload])

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        return sample_sources

    node = make_fundamentals_node(client, retrieve)
    update = await node(_state())

    findings = update["findings"]
    # The pure-fabrication finding is dropped; the mixed one keeps only valid cites.
    assert len(findings) == 1
    assert findings[0].cited_chunk_ids == (102,)


async def test_no_sources_means_no_findings() -> None:
    client = ScriptedLLMClient(responses=[])  # not consulted

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        return []

    node = make_fundamentals_node(client, retrieve)
    update = await node(_state())

    assert update["sources"] == []
    assert update["findings"] == []
    assert update["usage"] == []
    assert client.calls == []  # short-circuit before the LLM call


async def test_parse_failure_emits_no_findings(
    sample_sources: list[Source],
) -> None:
    client = ScriptedLLMClient(responses=["hmm not really json"])

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        return sample_sources

    node = make_fundamentals_node(client, retrieve)
    update = await node(_state())

    assert update["findings"] == []
    # Sources still flow downstream so the critic can act on the pool.
    assert len(update["sources"]) == 3
