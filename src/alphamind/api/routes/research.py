"""POST /research — agent-graph streaming endpoint.

The handler validates the request, builds a :class:`GraphInput`, and
returns a Server-Sent Event stream that emits one event per graph node
as it completes. The order is router → specialists (in completion
order, possibly parallel) → synthesizer → critic → ``done``.

The synthesizer's final answer arrives as a single ``thesis`` event,
not token-by-token. Token streaming requires growing the
:class:`alphamind.llm.base.LLMClient` protocol with a streaming
method; that lands in a follow-up.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from alphamind.agents.graph import ResearchGraph
from alphamind.agents.state import GraphInput
from alphamind.api.dependencies import get_research_graph_dep
from alphamind.api.schemas import ResearchRequest
from alphamind.api.sse import stream_research

router = APIRouter(tags=["research"])


@router.post(
    "/research",
    summary="Run the agent graph and stream progress as Server-Sent Events.",
    responses={
        200: {
            "description": (
                "text/event-stream of SSE events: router-decision, "
                "specialist-report (one per specialist), thesis, "
                "critic-report, then done."
            ),
            "content": {"text/event-stream": {}},
        }
    },
)
async def post_research(
    request: ResearchRequest,
    # ``Depends(...)`` in argument defaults is the FastAPI idiom — the
    # framework reads the call expression itself, not the result. B008
    # is the general "no function calls in defaults" rule and doesn't
    # know about FastAPI's signature inspection.
    graph: ResearchGraph = Depends(get_research_graph_dep),  # noqa: B008
) -> EventSourceResponse:
    payload = GraphInput(
        query=request.query,
        as_of=request.as_of,
        top_k=request.top_k,
    )
    # ``EventSourceResponse`` consumes the async iterator and writes
    # ``event:``/``data:`` framing for us. It also handles client
    # disconnects by cancelling the generator.
    return EventSourceResponse(stream_research(graph, payload))


__all__ = ["router"]
