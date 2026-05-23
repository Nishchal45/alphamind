"""Tests for the technical specialist's no-data placeholder.

The contract these tests pin down:

- The factory signature matches the other specialists, so the graph
  builder doesn't need a special case.
- The node never calls the retrieval function or the LLM.
- The node emits empty ``sources``, ``findings``, and ``usage`` —
  consistent with what the base pipeline would emit when retrieval
  comes up empty.
"""

from __future__ import annotations

from datetime import date

import pytest

from alphamind.agents.specialists.technical import (
    NO_DATA_REASON,
    NODE_NAME,
    make_technical_node,
)
from alphamind.agents.state import ResearchState, Source
from tests.unit.agents.conftest import ScriptedLLMClient

pytestmark = pytest.mark.asyncio


def _state() -> ResearchState:
    return ResearchState(
        query="NVDA momentum into year-end?",
        as_of=date(2024, 12, 31),
        top_k=3,
    )


async def test_node_emits_no_findings_and_skips_llm_and_retrieve() -> None:
    # ScriptedLLMClient with no responses will raise if ``complete`` is
    # called — exactly what we want to assert about the stub.
    client = ScriptedLLMClient(responses=[])
    retrieve_calls: list[tuple[str, date, int]] = []

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        retrieve_calls.append((query, as_of, top_k))
        return []

    node = make_technical_node(client, retrieve)
    update = await node(_state())

    assert update == {"sources": [], "findings": [], "usage": []}
    assert client.calls == []
    assert retrieve_calls == []


async def test_module_metadata_exposes_reason_and_node_name() -> None:
    assert NODE_NAME == "technical"
    assert "market-data adapter" in NO_DATA_REASON
