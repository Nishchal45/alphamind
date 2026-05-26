"""Fixtures for the API tests.

The strategy: build the real LangGraph DAG via ``build_research_graph``
but with a scripted LLM and a stub retrieval function. That way the
HTTP tests exercise the same wiring the production server uses —
routing, fan-out, citation validation, streaming — without touching
Postgres or Anthropic.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import date
from typing import Any

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from langgraph.graph.state import CompiledStateGraph

from alphamind.agents import build_research_graph
from alphamind.agents.state import Source
from alphamind.api.app import create_app
from tests.unit.agents.conftest import SystemKeyedLLMClient


def build_stub_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    """A compiled graph wired with a scripted LLM and a canned source pool.

    Mirrors the SystemKeyedLLMClient pattern from
    ``tests/unit/agents/test_graph.py``: each node's prompt contains a
    distinctive marker (``"routing layer"``, ``"fundamentals
    specialist"``, ``"synthesizer"``, ``"critic"``) and the client
    returns the canned reply that matches.
    """
    router_resp = json.dumps(
        {
            "primary": "fundamentals",
            "specialists": ["fundamentals", "risk"],
            "rationale": "Bull / bear question spans revenue mix and risks.",
        }
    )
    fundamentals_resp = json.dumps(
        {
            "findings": [
                {"claim": "China is 17% of revenue.", "cited_chunk_ids": [101]},
                {"claim": "Gross margin reached 73%.", "cited_chunk_ids": [102]},
            ]
        }
    )
    synthesizer_resp = json.dumps(
        {
            "summary": "Margins are strong, but China concentration is a risk.",
            "bull_case": [
                {"claim": "Gross margin expanded.", "cited_chunk_ids": [102]},
            ],
            "bear_case": [
                {"claim": "China revenue concentration.", "cited_chunk_ids": [101]},
            ],
        }
    )
    critic_resp = json.dumps({"issues": []})

    client = SystemKeyedLLMClient(
        responses_by_marker={
            "routing layer": router_resp,
            "fundamentals specialist": fundamentals_resp,
            "risk specialist": json.dumps({"findings": []}),
            "synthesizer": synthesizer_resp,
            "critic": critic_resp,
        }
    )

    sources = [
        Source(
            chunk_id=101,
            filing_id=1,
            ticker="NVDA",
            form="10-K",
            filing_date=date(2024, 2, 1),
            section="Item 1A",
            text="Sales to customers in China represented 17% of total revenue.",
            score=0.91,
        ),
        Source(
            chunk_id=102,
            filing_id=1,
            ticker="NVDA",
            form="10-K",
            filing_date=date(2024, 2, 1),
            section="Item 7",
            text="Gross margin expanded to 73% as Data Center mix increased.",
            score=0.84,
        ),
    ]

    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        return sources

    return build_research_graph(llm=client, retrieve=retrieve)


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """An ``httpx.AsyncClient`` bound to a fresh FastAPI app with a stub graph.

    The ``async with`` on the client triggers the lifespan handler so
    ``app.state.graph`` is set before the test issues its first
    request.
    """
    app = create_app(graph=build_stub_graph())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
