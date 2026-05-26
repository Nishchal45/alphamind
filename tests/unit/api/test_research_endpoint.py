"""``POST /research`` SSE streaming tests.

The transport-level contract: response is ``text/event-stream``;
events are ``\\n\\n``-separated blocks of ``field: value`` lines. We
parse them manually rather than wiring an SSE client library, because
the shape is small enough that a 15-line parser is clearer than an
extra dependency.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


def _parse_sse(payload: str) -> list[dict[str, str]]:
    """Parse an SSE response body into a list of ``{event, data}`` dicts.

    Comments (``:`` lines, which sse-starlette uses for keepalive pings)
    and unknown fields are dropped. Multi-line ``data:`` is joined with
    newlines per the SSE spec, although our adapter only ever emits
    single-line JSON.
    """
    # sse-starlette uses ``\r\n`` line endings per the SSE spec; normalise
    # so the splitter sees the canonical ``\n\n`` event separator.
    normalised = payload.replace("\r\n", "\n")
    events: list[dict[str, str]] = []
    for raw_block in normalised.split("\n\n"):
        block = raw_block.strip("\n")
        if not block:
            continue
        current: dict[str, str] = {}
        data_lines: list[str] = []
        for raw_line in block.split("\n"):
            line = raw_line.rstrip("\r")
            if not line or line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value.lstrip(" ")
            if field == "data":
                data_lines.append(value)
            elif field == "event":
                current["event"] = value
        if data_lines:
            current["data"] = "\n".join(data_lines)
        if current:
            events.append(current)
    return events


def _events_by_type(events: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    """Group parsed SSE events by their ``event`` field, decoding data as JSON."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for evt in events:
        kind = evt.get("event", "message")
        payload = json.loads(evt["data"]) if "data" in evt else {}
        grouped.setdefault(kind, []).append(payload)
    return grouped


async def test_research_streams_router_specialist_synthesizer_critic_done(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/research",
        json={
            "query": "What is NVDA's China exposure?",
            "as_of": "2024-12-31",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    grouped = _events_by_type(events)

    # Every super-step lands as a node event; the stub graph fires
    # router → fundamentals → risk (parallel) → synthesizer → critic.
    node_payloads = grouped.get("node", [])
    nodes_seen = {p["node"] for p in node_payloads}
    assert nodes_seen == {"router", "fundamentals", "risk", "synthesizer", "critic"}

    # End-of-stream signal must always land.
    assert "done" in grouped
    assert grouped["done"] == [{}]

    # Synthesizer event carries the thesis.
    synth = next(p for p in node_payloads if p["node"] == "synthesizer")
    thesis = synth["payload"]["thesis"]
    assert "summary" in thesis
    assert thesis["bull_case"][0]["cited_chunk_ids"] == [102]
    assert thesis["bear_case"][0]["cited_chunk_ids"] == [101]


async def test_research_validates_required_fields(client: AsyncClient) -> None:
    response = await client.post("/research", json={"query": "anything"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    fields_with_errors = {tuple(err["loc"]) for err in detail}
    assert ("body", "as_of") in fields_with_errors


async def test_research_rejects_empty_query(client: AsyncClient) -> None:
    response = await client.post(
        "/research",
        json={"query": "", "as_of": "2024-12-31"},
    )
    assert response.status_code == 422


async def test_research_rejects_top_k_out_of_range(client: AsyncClient) -> None:
    response = await client.post(
        "/research",
        json={"query": "q", "as_of": "2024-12-31", "top_k": 999},
    )
    assert response.status_code == 422


async def test_research_emits_error_event_when_stream_raises() -> None:
    """A graph that raises mid-stream should land as a structured ``error`` event.

    Builds a fresh ``create_app`` with a graph that explodes inside
    ``astream`` so the failure path runs end-to-end through SSE.
    """
    from httpx import ASGITransport, AsyncClient  # noqa: PLC0415

    from alphamind.api.app import create_app  # noqa: PLC0415

    class ExplodingGraph:
        async def astream(self, *_args, **_kwargs):
            raise RuntimeError("boom from the depths")
            yield  # pragma: no cover - unreachable, makes this an async generator

    app = create_app(graph=ExplodingGraph())  # type: ignore[arg-type]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.post(
            "/research",
            json={"query": "q", "as_of": "2024-12-31"},
        )
    assert response.status_code == 200  # stream opened before the failure

    events = _parse_sse(response.text)
    grouped = _events_by_type(events)
    assert "error" in grouped
    payload = grouped["error"][0]
    assert payload["type"] == "RuntimeError"
    assert "boom from the depths" in payload["error"]
    # ``done`` must not land when the stream errored.
    assert "done" not in grouped
