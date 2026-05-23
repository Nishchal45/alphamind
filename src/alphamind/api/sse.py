"""Convert agent-graph updates into Server-Sent Event payloads.

LangGraph emits ``{node_name: state_update}`` pairs via
``CompiledStateGraph.astream(stream_mode="updates")``. SSE wants
``{event, data}`` dicts where ``data`` is a JSON string. This module
is the seam — kept separate from the route handler so the conversion
logic can be unit-tested without spinning up FastAPI.

Event types (kebab-cased to match HTTP conventions; the ``event:``
header is what JavaScript ``EventSource`` keys handlers off):

- ``router-intent`` — fired once after the router runs.
- ``specialist-findings`` — fired once per specialist that completed.
  Specialists run in parallel; LangGraph emits them in completion
  order, not declaration order.
- ``thesis`` — fired once after the synthesizer.
- ``critique`` — fired once after the critic.
- ``error`` — terminal; the graph raised.
- ``done`` — terminal sentinel for a clean run.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from alphamind.agents.state import (
    Critique,
    Finding,
    ResearchState,
    RouterIntent,
    Source,
    Thesis,
    Usage,
)
from alphamind.api.schemas import (
    CritiqueIssuePayload,
    CritiquePayload,
    ErrorPayload,
    FindingPayload,
    RouterIntentPayload,
    SourcePayload,
    SpecialistEventPayload,
    ThesisPayload,
    UsagePayload,
)

logger = logging.getLogger(__name__)

SSEEvent = dict[str, str]

_SPECIALIST_NODES: frozenset[str] = frozenset({"fundamentals", "sentiment", "risk", "technical"})


def _event(name: str, payload: Any) -> SSEEvent:
    """Build an SSE event dict.

    ``sse-starlette`` reads ``event`` and ``data`` keys. ``data`` must
    be a string; pydantic models JSON-encode via ``model_dump_json()``.
    """

    return {
        "event": name,
        "data": (
            payload.model_dump_json() if hasattr(payload, "model_dump_json") else str(payload)
        ),
    }


def _events_for_update(node_name: str, state_update: dict[str, Any]) -> Iterable[SSEEvent]:
    """Yield zero or more SSE events for one LangGraph node update."""

    # Router writes ``intent`` (+ ``usage``).
    intent = state_update.get("intent")
    if isinstance(intent, RouterIntent):
        yield _event("router-intent", RouterIntentPayload.from_dataclass(intent))

    # Specialists write ``sources`` + ``findings`` (+ ``usage``). Group
    # them into one specialist event keyed by the LangGraph node name —
    # client code doesn't have to correlate three separate streams to
    # know which specialist emitted what.
    if node_name in _SPECIALIST_NODES:
        sources_raw = state_update.get("sources", [])
        findings_raw = state_update.get("findings", [])
        usage_raw = state_update.get("usage", [])
        if sources_raw or findings_raw or usage_raw:
            yield _event(
                "specialist-findings",
                SpecialistEventPayload(
                    specialist=node_name,  # type: ignore[arg-type]
                    sources=[
                        SourcePayload.from_dataclass(s)
                        for s in sources_raw
                        if isinstance(s, Source)
                    ],
                    findings=[
                        FindingPayload.from_dataclass(f)
                        for f in findings_raw
                        if isinstance(f, Finding)
                    ],
                    usage=[
                        UsagePayload.from_dataclass(u) for u in usage_raw if isinstance(u, Usage)
                    ],
                ),
            )

    # Synthesizer writes ``thesis`` (+ ``usage``).
    thesis = state_update.get("thesis")
    if isinstance(thesis, Thesis):
        yield _event("thesis", ThesisPayload.from_dataclass(thesis))

    # Critic writes ``critique`` (+ ``usage``).
    critique = state_update.get("critique")
    if isinstance(critique, Critique):
        yield _event(
            "critique",
            CritiquePayload(
                issues=[CritiqueIssuePayload.from_dataclass(i) for i in critique.issues],
                parse_error=critique.parse_error,
            ),
        )


async def stream_research(
    graph: CompiledStateGraph[ResearchState, Any, Any, Any],
    state: ResearchState,
) -> AsyncIterator[SSEEvent]:
    """Drive the graph and yield SSE events.

    On unexpected failure, emits a final ``error`` event with the
    message and exits cleanly. Per-node failures are already handled
    inside the graph (each node has a fallback path); this catch is
    for the graph itself raising — which currently only happens for
    wiring bugs that escape input validation at the route layer.
    """

    try:
        async for update in graph.astream(state, stream_mode="updates"):
            for node_name, state_update in update.items():
                for event in _events_for_update(node_name, state_update):
                    yield event
    except Exception as exc:  # final-event handler
        logger.exception("research graph raised")
        yield _event("error", ErrorPayload(message=str(exc)))
        return

    yield {"event": "done", "data": "{}"}


__all__ = ["SSEEvent", "stream_research"]
