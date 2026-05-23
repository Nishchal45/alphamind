"""Tests for the critic node."""

from __future__ import annotations

import json

import pytest

from alphamind.agents.critic import make_critic_node
from alphamind.agents.state import ResearchState, Source, Thesis, ThesisClaim
from tests.unit.agents.conftest import ScriptedLLMClient

pytestmark = pytest.mark.asyncio


def _state_with_thesis(sources: list[Source]) -> ResearchState:
    thesis = Thesis(
        summary="Strong margins, geopolitical risk.",
        bull_case=(ThesisClaim(claim="Margins expanded.", cited_chunk_ids=(102,)),),
        bear_case=(ThesisClaim(claim="China concentration is material.", cited_chunk_ids=(101,)),),
    )
    return ResearchState(
        query="NVDA bull / bear?",
        thesis=thesis,
        sources=sources,
    )


async def test_parses_critic_issues(sample_sources: list[Source]) -> None:
    payload = json.dumps(
        {
            "issues": [
                {
                    "kind": "unsupported",
                    "claim": "Margins expanded.",
                    "detail": (
                        "Chunk 102 talks about gross margin "
                        "but the bull claim implies operating margin."
                    ),
                    "cited_chunk_ids": [102],
                }
            ]
        }
    )
    client = ScriptedLLMClient(responses=[payload])
    node = make_critic_node(client)

    update = await node(_state_with_thesis(sample_sources))
    issues = update["critique"].issues
    assert len(issues) == 1
    assert issues[0].kind == "unsupported"
    assert issues[0].cited_chunk_ids == (102,)
    assert update["critique"].parse_error is None


async def test_parse_failure_records_error(sample_sources: list[Source]) -> None:
    client = ScriptedLLMClient(responses=["not json at all"])
    node = make_critic_node(client)

    update = await node(_state_with_thesis(sample_sources))
    critique = update["critique"]
    assert critique.issues == ()
    assert critique.parse_error is not None and "not json" in critique.parse_error


async def test_empty_thesis_skips_llm_call(sample_sources: list[Source]) -> None:
    client = ScriptedLLMClient(responses=[])
    node = make_critic_node(client)

    state = ResearchState(
        query="anything",
        thesis=Thesis(summary="(none)", bull_case=(), bear_case=()),
        sources=sample_sources,
    )
    update = await node(state)
    assert update["critique"].issues == ()
    assert update["critique"].parse_error is None
    assert client.calls == []


async def test_empty_source_pool_flags_thesis() -> None:
    client = ScriptedLLMClient(responses=[])
    node = make_critic_node(client)

    update = await node(_state_with_thesis(sources=[]))
    issues = update["critique"].issues
    assert len(issues) == 1
    assert issues[0].kind == "unsupported"
    assert client.calls == []
