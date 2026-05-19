"""Convert agent-graph updates into Server-Sent Event payloads.

LangGraph emits ``(node_name, state_update)`` pairs as each node
completes. The SSE protocol wants ``{event, data}`` dicts where
``data`` is a string (JSON-encoded). This module is the seam between
the two — kept separate from the route handler so the conversion
logic can be unit-tested without spinning up FastAPI.

Event types (kebab-cased to match HTTP conventions; the ``event:``
header is what JavaScript ``EventSource`` keys handlers off):

- ``router-decision`` — fired once after the router runs.
- ``specialist-report`` — fired once per specialist, in completion
  order (parallel branches resolve in the order they finish).
- ``thesis`` — fired once after the synthesizer.
- ``critic-report`` — fired once after the critic.
- ``error`` — terminal; the graph raised.
- ``done`` — terminal sentinel so clients know the stream is closed
  cleanly. Useful for distinguishing "stream ended" from "connection
  dropped."
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any

from alphamind.agents.graph import ResearchGraph
from alphamind.agents.state import (
    CriticReport,
    GraphInput,
    RouterDecision,
    SpecialistReport,
    Thesis,
)
from alphamind.api.schemas import (
    CriticReportPayload,
    ErrorPayload,
    RouterDecisionPayload,
    SpecialistReportPayload,
    ThesisPayload,
)

logger = logging.getLogger(__name__)


SSEEvent = dict[str, str]


def _event(name: str, payload: Any) -> SSEEvent:
    """Build a Server-Sent Event dict.

    ``sse-starlette`` reads ``event`` and ``data`` keys. ``data`` must
    be a string; pydantic models JSON-encode via ``model_dump_json()``.
    """

    return {
        "event": name,
        "data": payload.model_dump_json() if hasattr(payload, "model_dump_json") else str(payload),
    }


def _events_for_update(node_name: str, state_update: dict[str, Any]) -> Iterable[SSEEvent]:
    """Yield zero or more SSE events for one LangGraph state-update."""

    # Router writes ``router_decision``.
    decision = state_update.get("router_decision")
    if isinstance(decision, RouterDecision):
        yield _event("router-decision", RouterDecisionPayload.from_dataclass(decision))

    # Specialists append to ``specialist_reports`` (one item per emit).
    reports = state_update.get("specialist_reports")
    if isinstance(reports, list):
        for report in reports:
            if isinstance(report, SpecialistReport):
                yield _event(
                    "specialist-report",
                    SpecialistReportPayload.from_dataclass(report),
                )

    # Synthesizer writes ``thesis``.
    thesis = state_update.get("thesis")
    if isinstance(thesis, Thesis):
        yield _event("thesis", ThesisPayload.from_dataclass(thesis))

    # Critic writes ``critic_report``.
    critic = state_update.get("critic_report")
    if isinstance(critic, CriticReport):
        yield _event("critic-report", CriticReportPayload.from_dataclass(critic))

    # Node names that emit nothing (or only fields we already handled)
    # are silently swallowed. Keep them noted for debugging.
    if not any(
        key in state_update
        for key in ("router_decision", "specialist_reports", "thesis", "critic_report")
    ):
        logger.debug("no SSE event derived from node %r update %r", node_name, state_update)


async def stream_research(
    graph: ResearchGraph,
    payload: GraphInput,
) -> AsyncIterator[SSEEvent]:
    """Drive the graph and yield SSE events.

    On unexpected failure, emits a final ``error`` event with the
    message and exits cleanly. Individual node failures are already
    handled inside the graph (each node has a fallback path); this
    catch is for the *graph itself* raising — which currently only
    happens for input-validation bugs that escape the route layer.
    """

    try:
        async for node_name, state_update in graph.stream_updates(payload):
            for event in _events_for_update(node_name, state_update):
                yield event
    except Exception as exc:
        logger.exception("research graph raised")
        yield _event("error", ErrorPayload(message=str(exc)))
        return

    yield {"event": "done", "data": "{}"}


__all__ = ["SSEEvent", "stream_research"]
