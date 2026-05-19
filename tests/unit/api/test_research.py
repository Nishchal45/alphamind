"""Tests for POST /research.

These tests run the FastAPI app in-process via httpx ``ASGITransport``
and replace the agent graph with :class:`StubResearchGraph` so the
HTTP / SSE layer is the only thing under test. Graph correctness is
covered in :mod:`tests.unit.agents`.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

from alphamind.agents.state import (
    Citation,
    Claim,
    CriticReport,
    RouterDecision,
    SpecialistReport,
    Thesis,
    UnsupportedClaim,
)
from alphamind.api.app import create_app
from alphamind.api.dependencies import get_research_graph_dep
from tests.unit.api.conftest import build_app

pytestmark = pytest.mark.asyncio


def _citation(ordinal: int, chunk_id: int) -> Citation:
    return Citation(
        ordinal=ordinal,
        chunk_id=chunk_id,
        filing_id=chunk_id * 10,
        ticker="NVDA",
        form="10-K",
        filing_date=date(2024, 1, 1),
        section="Item 7",
        text=f"source-{chunk_id}",
    )


def _parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE response body into ``(event, data)`` pairs.

    SSE framing: a record is one or more ``field: value`` lines
    separated by blank lines. ``data`` lines concatenate. ``event``
    defaults to ``message`` when absent.
    """

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
        if line.startswith(":"):  # SSE comments
            continue
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())

    _flush()
    return events


async def test_research_streams_all_node_events(asgi_transport_factory: Any) -> None:
    citation = _citation(1, 42)
    router_update = {
        "router_decision": RouterDecision(specialists=("fundamentals",), rationale="r"),
    }
    updates: list[tuple[str, dict[str, Any]]] = [
        ("router", router_update),
        (
            "fundamentals",
            {
                "specialist_reports": [
                    SpecialistReport(
                        specialist="fundamentals",
                        summary="data center is huge",
                        claims=(Claim(text="DC grew 86%", citations=(1,)),),
                        citations=(citation,),
                    )
                ]
            },
        ),
        (
            "synthesizer",
            {
                "thesis": Thesis(
                    answer="DC growth dominates [1].",
                    bull=("DC grew 86% [1].",),
                    bear=(),
                    citations=(citation,),
                )
            },
        ),
        (
            "critic",
            {
                "critic_report": CriticReport(
                    unsupported=(UnsupportedClaim(claim="x", reason="y"),),
                    notes="one flag",
                )
            },
        ),
    ]
    app, stub = build_app(updates)

    async with asgi_transport_factory(app) as client:
        response = await client.post(
            "/research",
            json={"query": "NVDA bull/bear", "as_of": "2024-12-31", "top_k": 5},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    event_names = [name for name, _ in events]
    assert event_names == [
        "router-decision",
        "specialist-report",
        "thesis",
        "critic-report",
        "done",
    ]

    # Payloads carry the full state — router-decision has the rationale,
    # specialist-report carries claims + citations, thesis has bull/bear,
    # critic-report flags ok=False because there's an unsupported claim.
    assert events[0][1]["rationale"] == "r"
    assert events[1][1]["specialist"] == "fundamentals"
    assert events[1][1]["claims"][0]["text"] == "DC grew 86%"
    assert events[2][1]["bull"] == ["DC grew 86% [1]."]
    assert events[3][1]["ok"] is False

    # The stub captured the GraphInput we sent.
    assert stub.last_payload is not None
    assert stub.last_payload.query == "NVDA bull/bear"
    assert stub.last_payload.as_of == date(2024, 12, 31)
    assert stub.last_payload.top_k == 5


async def test_research_rejects_blank_query(asgi_transport_factory: Any) -> None:
    app, _ = build_app([])
    async with asgi_transport_factory(app) as client:
        response = await client.post(
            "/research",
            json={"query": "   ", "as_of": "2024-12-31"},
        )

    # Pydantic field validator should reject whitespace-only queries.
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

    async def stream_updates(self, _payload: Any) -> Any:
        yield (
            "router",
            {"router_decision": RouterDecision(specialists=("fundamentals",), rationale="r")},
        )
        raise RuntimeError("boom")


def _exploding_graph() -> _ExplodingGraph:
    return _ExplodingGraph()


async def test_research_emits_error_event_when_graph_raises(
    asgi_transport_factory: Any,
) -> None:
    # An empty updates list means the stub yields nothing — but the
    # graph runs to completion successfully. For an error path we need
    # a stub that raises mid-stream.
    app = create_app()
    app.dependency_overrides[get_research_graph_dep] = _exploding_graph

    async with asgi_transport_factory(app) as client:
        response = await client.post(
            "/research",
            json={"query": "ok", "as_of": "2024-12-31"},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    event_names = [name for name, _ in events]
    # Router event makes it out before the failure, then the error
    # event lands and the stream ends without a 'done' sentinel.
    assert event_names[0] == "router-decision"
    assert "error" in event_names
    assert "done" not in event_names
    error_event = next(payload for name, payload in events if name == "error")
    assert "boom" in error_event["message"]
