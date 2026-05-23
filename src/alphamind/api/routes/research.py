"""POST /research — agent-graph streaming endpoint.

The handler validates the request, builds a :class:`ResearchState`
seed, and returns a Server-Sent Event stream that emits one event per
graph node as it completes. Order is router → specialists (in
completion order, potentially parallel) → synthesizer → critic → done
(or error if the graph itself raised).

The synthesizer's thesis arrives as a single ``thesis`` event — token
streaming requires growing :class:`alphamind.llm.base.LLMClient` and
ships in a follow-up.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from alphamind.agents.state import ResearchState
from alphamind.api.dependencies import ResearchGraphT, get_research_graph_dep
from alphamind.api.schemas import ResearchRequest
from alphamind.api.sse import stream_research

router = APIRouter(tags=["research"])


@router.post(
    "/research",
    summary="Run the agent graph and stream progress as Server-Sent Events.",
    responses={
        200: {
            "description": (
                "text/event-stream of SSE events: router-intent, "
                "specialist-findings (one per specialist), thesis, "
                "critique, then done."
            ),
            "content": {"text/event-stream": {}},
        }
    },
)
async def post_research(
    request: ResearchRequest,
    # ``Depends(...)`` in argument defaults is the FastAPI idiom; ruff's
    # B008 (no function calls in defaults) doesn't know about FastAPI's
    # signature inspection.
    graph: ResearchGraphT = Depends(get_research_graph_dep),  # noqa: B008
) -> EventSourceResponse:
    state: ResearchState = {
        "query": request.query,
        "as_of": request.as_of,
        "top_k": request.top_k,
    }
    return EventSourceResponse(stream_research(graph, state))


__all__ = ["router"]
