"""SSE adapter for the LangGraph research DAG.

:func:`stream_research_events` consumes ``graph.astream(mode="updates")``
and yields one SSE event per super-step. Each event's ``data`` field is
JSON encoding the partial state update returned by the node that just
completed.

Three event types are emitted:

- ``node`` — one per LangGraph update, payload is
  ``{"node": <name>, "payload": <jsonable dict>}``.
- ``done`` — synthetic end-of-stream signal, payload is ``{}``. SSE has
  no built-in stream terminator; clients shouldn't have to guess from
  silence whether the run finished or the connection dropped.
- ``error`` — emitted when any exception leaks out of the stream,
  payload is ``{"error": <message>, "type": <exception class name>}``.
  The stream then closes cleanly so the client can decide whether to
  retry.

State is composed of frozen dataclasses (``Source``, ``Finding``,
``Thesis``, ``Critique``…); the JSON encoder walks them via
``dataclasses.asdict`` and serialises ``date`` / ``datetime`` as ISO
8601. Tuples become lists.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any

from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


def _to_jsonable(obj: Any) -> Any:
    """Convert a value tree containing dataclasses and dates to JSON-safe form.

    Recursive: dicts, lists, tuples, dataclass instances. Anything
    unknown is returned as-is and lets ``json.dumps`` raise so we
    catch the encoding bug instead of silently emitting garbage.
    """
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(x) for x in obj]
    return obj


async def stream_research_events(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    *,
    query: str,
    as_of: date,
    top_k: int,
) -> AsyncIterator[dict[str, str]]:
    """Yield SSE-shaped events for one research run.

    The dict shape matches what ``sse_starlette.EventSourceResponse``
    expects: ``{"event": <name>, "data": <serialized string>}``.
    """
    state_input: dict[str, Any] = {"query": query, "as_of": as_of, "top_k": top_k}

    try:
        async for update in graph.astream(state_input, stream_mode="updates"):
            # In "updates" mode, each yield is {node_name: partial_state}.
            # Multiple super-steps can land in one yield when LangGraph
            # fans out, so iterate.
            for node_name, partial in update.items():
                payload = _to_jsonable(partial)
                yield {
                    "event": "node",
                    "data": json.dumps({"node": node_name, "payload": payload}),
                }
        yield {"event": "done", "data": "{}"}
    except Exception as exc:  # fail-soft into the SSE channel (broad catch is intentional)
        # The connection is mid-stream; we can't return a 500 anymore.
        # Surface the failure as a structured event and let the client
        # decide whether to retry. Logged at warning rather than error
        # because client cancellation is a normal cause.
        logger.warning("research stream aborted: %s", exc, exc_info=True)
        yield {
            "event": "error",
            "data": json.dumps(
                {"error": str(exc), "type": type(exc).__name__},
            ),
        }


__all__ = ["stream_research_events"]
