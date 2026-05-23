"""Tests for POST /research.

These run the FastAPI app in-process via httpx ``ASGITransport`` and
inject a :class:`StubResearchGraph` that yields canned LangGraph
update events. Graph correctness lives in :mod:`tests.unit.agents`;
this file is just the HTTP / SSE surface.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

from alphamind.agents.state import (
    Critique,
    CritiqueIssue,
    Finding,
    RouterIntent,
    Source,
    Thesis,
    ThesisClaim,
    Usage,
)
from alphamind.api.app import create_app
from alphamind.api.dependencies import get_research_graph_dep
from tests.unit.api.conftest import build_app

pytestmark = pytest.mark.asyncio


def _source(chunk_id: int = 101) -> Source:
    return Source(
        chunk_id=chunk_id,
        filing_id=1,
        ticker="NVDA",
        form="10-K",
        filing_date=date(2024, 2, 1),
        section="Item 1A",
        text="China sales are concentrated.",
        score=0.91,
    )


def _parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE response body into ``(event, data)`` pairs."""

    events: list[tuple[str, dict[str, Any]]] = []
    current_event: str | None = None
    data_lines: list[str] = []

    def _flush() -> None:
        if current_event is None and not data_lines:
            return
        raw = "\n".join(data_lines).strip()
        payload: dict[str, Any] = json.loads(raw) if raw else {}
        events.append((current_event or "message", payload))

    for line in body.splitlines():
        if not line.strip():
            _flush()
            current_event = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())

    _flush()
    return events


async def test_research_streams_all_node_events(asgi_transport_factory: Any) -> None:
    intent = RouterIntent(
        primary="fundamentals",
        specialists=("fundamentals",),
        rationale="revenue mix question",
    )
    finding = Finding(
        specialist="fundamentals",
        claim="China is 17% of revenue.",
        cited_chunk_ids=(101,),
    )
    usage_fund = Usage(node="fundamentals", model="m", input_tokens=100, output_tokens=20)
    thesis = Thesis(
        summary="Margins strong; China concentration.",
        bull_case=(ThesisClaim(claim="margin", cited_chunk_ids=(101,)),),
        bear_case=(ThesisClaim(claim="concentration", cited_chunk_ids=(101,)),),
    )
    critique = Critique(
        issues=(
            CritiqueIssue(
                kind="unsupported",
                claim="margin",
                detail="weak support",
                cited_chunk_ids=(101,),
            ),
        ),
        parse_error=None,
    )

    updates = [
        {
            "router": {
                "intent": intent,
                "usage": [Usage(node="router", model="m", input_tokens=50, output_tokens=10)],
            }
        },
        {
            "fundamentals": {
                "sources": [_source(101)],
                "findings": [finding],
                "usage": [usage_fund],
            }
        },
        {
            "synthesizer": {
                "thesis": thesis,
                "usage": [Usage(node="synthesizer", model="m", input_tokens=200, output_tokens=50)],
            }
        },
        {
            "critic": {
                "critique": critique,
                "usage": [Usage(node="critic", model="m", input_tokens=300, output_tokens=30)],
            }
        },
    ]
    app, stub = build_app(updates)

    async with asgi_transport_factory(app) as client:
        response = await client.post(
            "/research",
            json={"query": "NVDA China?", "as_of": "2024-12-31", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    names = [n for n, _ in events]
    assert names == [
        "router-intent",
        "specialist-findings",
        "thesis",
        "critique",
        "done",
    ]

    # Payload spot-checks.
    assert events[0][1]["rationale"] == "revenue mix question"
    spec_payload = events[1][1]
    assert spec_payload["specialist"] == "fundamentals"
    assert spec_payload["findings"][0]["claim"] == "China is 17% of revenue."
    assert spec_payload["sources"][0]["chunk_id"] == 101
    assert events[2][1]["summary"].startswith("Margins")
    assert events[3][1]["issues"][0]["kind"] == "unsupported"
    assert events[3][1]["parse_error"] is None

    # Stub captured the seed state.
    assert stub.last_stream_mode == "updates"
    assert stub.last_state == {"query": "NVDA China?", "as_of": date(2024, 12, 31), "top_k": 3}


async def test_research_rejects_blank_query(asgi_transport_factory: Any) -> None:
    app, _ = build_app([])
    async with asgi_transport_factory(app) as client:
        response = await client.post(
            "/research",
            json={"query": "   ", "as_of": "2024-12-31"},
        )
    assert response.status_code == 422


async def test_research_rejects_missing_as_of(asgi_transport_factory: Any) -> None:
    app, _ = build_app([])
    async with asgi_transport_factory(app) as client:
        response = await client.post("/research", json={"query": "ok"})
    assert response.status_code == 422


async def test_research_rejects_negative_top_k(asgi_transport_factory: Any) -> None:
    app, _ = build_app([])
    async with asgi_transport_factory(app) as client:
        response = await client.post(
            "/research",
            json={"query": "ok", "as_of": "2024-12-31", "top_k": 0},
        )
    assert response.status_code == 422


class _ExplodingGraph:
    """Yields one event then raises — drives the SSE error-event path."""

    async def astream(
        self,
        _state: dict[str, Any],
        *,
        stream_mode: str = "values",
    ) -> Any:
        # ``stream_mode`` is part of the contract; pin it so we don't
        # silently switch and surprise the SSE adapter.
        assert stream_mode == "updates"
        yield {
            "router": {
                "intent": RouterIntent(
                    primary="fundamentals",
                    specialists=("fundamentals",),
                    rationale="r",
                ),
                "usage": [],
            }
        }
        raise RuntimeError("boom")


def _exploding_graph() -> _ExplodingGraph:
    return _ExplodingGraph()


async def test_research_emits_error_event_when_graph_raises(
    asgi_transport_factory: Any,
) -> None:
    app = create_app()
    app.dependency_overrides[get_research_graph_dep] = _exploding_graph

    async with asgi_transport_factory(app) as client:
        response = await client.post(
            "/research",
            json={"query": "ok", "as_of": "2024-12-31"},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    names = [n for n, _ in events]
    assert names[0] == "router-intent"
    assert "error" in names
    assert "done" not in names
    err = next(payload for n, payload in events if n == "error")
    assert "boom" in err["message"]
