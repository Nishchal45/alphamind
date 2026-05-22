"""LangGraph DAG for the research pipeline.

The vertical slice wires:

    START → router → fundamentals → synthesizer → critic → END

The router → specialist edge is conditional so the graph can fan out
to additional specialists as they land (sentiment, technical, risk).
For now the routing function always selects ``fundamentals`` because
that's the only specialist node compiled in.

The factory takes a :class:`LLMClient` and a retrieval function and
returns a compiled graph. Tests inject fakes; production wires the
real Anthropic client and a SQLAlchemy-backed retrieval entry-point.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from alphamind.agents.critic import make_critic_node
from alphamind.agents.router import make_router_node
from alphamind.agents.specialists.fundamentals import RetrievalFn, make_fundamentals_node
from alphamind.agents.state import ResearchState
from alphamind.agents.synthesizer import make_synthesizer_node
from alphamind.llm import LLMClient

# Re-export so callers can import a single name for the retrieval contract.
__all__ = ["RetrievalFn", "build_research_graph"]


def _route_specialists(state: ResearchState) -> Literal["fundamentals"]:
    """Dispatch from the router to one or more specialist nodes.

    Today the graph only contains ``fundamentals``; the router's
    ``specialists`` list is recorded on state for the synthesizer to
    reason about, but doesn't change the topology yet. When sentiment /
    technical / risk specialists land, the return type widens and this
    becomes a fan-out.
    """
    return "fundamentals"


def build_research_graph(
    *,
    llm: LLMClient,
    retrieve: RetrievalFn,
    router_model: str | None = None,
    specialist_model: str | None = None,
    synthesizer_model: str | None = None,
    critic_model: str | None = None,
) -> CompiledStateGraph[ResearchState, Any, Any, Any]:
    """Construct and compile the research DAG.

    Parameters
    ----------
    llm:
        Shared LLM client used by every node. Per-node model overrides
        let callers run, say, a cheaper model at the router and a
        more capable one at the critic.
    retrieve:
        Retrieval entry-point for the fundamentals specialist. See
        :data:`alphamind.agents.specialists.fundamentals.RetrievalFn`.
    """
    router = make_router_node(llm, model=router_model)
    fundamentals = make_fundamentals_node(llm, retrieve, model=specialist_model)
    synthesizer = make_synthesizer_node(llm, model=synthesizer_model)
    critic = make_critic_node(llm, model=critic_model)

    graph: StateGraph[ResearchState, Any, Any, Any] = StateGraph(ResearchState)
    # LangGraph's node-callable overloads don't unify with our async-node
    # signature under strict mypy. Suppression is scoped to these four
    # lines; nothing else in the agent layer needs the escape hatch.
    graph.add_node("router", _as_node(router))  # type: ignore[call-overload]
    graph.add_node("fundamentals", _as_node(fundamentals))  # type: ignore[call-overload]
    graph.add_node("synthesizer", _as_node(synthesizer))  # type: ignore[call-overload]
    graph.add_node("critic", _as_node(critic))  # type: ignore[call-overload]

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _route_specialists,
        {"fundamentals": "fundamentals"},
    )
    graph.add_edge("fundamentals", "synthesizer")
    graph.add_edge("synthesizer", "critic")
    graph.add_edge("critic", END)

    return graph.compile()


def _as_node(
    fn: Callable[[ResearchState], Awaitable[dict[str, Any]]],
) -> Callable[[ResearchState], Awaitable[dict[str, Any]]]:
    """Identity wrapper that gives LangGraph the exact signature it expects.

    Keeping a single explicit indirection here lets us bolt on tracing
    or retry decorators later without touching every node module.
    """
    return fn
