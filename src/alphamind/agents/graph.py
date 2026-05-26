"""LangGraph DAG for the research pipeline.

Topology:

::

    START → router → ┬─ fundamentals ─┐
                     ├─ risk ─────────┤
                     ├─ sentiment ────┤
                     └─ technical ────┴─→ synthesizer → critic → END

The router → specialist edge is a real fan-out: ``_route_specialists``
returns the list of specialist nodes to run in parallel based on the
intent the router produced. LangGraph fans in at the synthesizer — all
selected specialists must complete before synthesis runs.

``fundamentals`` is always included in the fan-out as a baseline, even
if the router omits it. The other specialists are gated by intent so a
purely fundamentals-shaped question doesn't pay for four LLM calls
when one is enough.

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
from alphamind.agents.specialists._base import RetrievalFn
from alphamind.agents.specialists.fundamentals import make_fundamentals_node
from alphamind.agents.specialists.risk import make_risk_node
from alphamind.agents.specialists.sentiment import make_sentiment_node
from alphamind.agents.specialists.technical import make_technical_node
from alphamind.agents.state import ResearchState
from alphamind.agents.synthesizer import make_synthesizer_node
from alphamind.llm import LLMClient

# Specialists wired into the graph today. Adding a new one means:
# (a) implementing the node, (b) appending the name here, (c) extending
# the path_map on add_conditional_edges, and (d) calling add_edge from
# the new node to "synthesizer".
_WIRED_SPECIALISTS: frozenset[str] = frozenset(
    {"fundamentals", "risk", "sentiment", "technical"}
)

# Re-export so callers can import a single name for the retrieval contract.
__all__ = ["RetrievalFn", "build_research_graph"]


_RoutedSpecialist = Literal["fundamentals", "risk", "sentiment", "technical"]


def _route_specialists(state: ResearchState) -> list[_RoutedSpecialist]:
    """Dispatch from the router to one or more specialist nodes.

    Reads ``state['intent'].specialists`` and returns the subset that
    matches a wired specialist node, in a stable order. ``fundamentals``
    is forced into the list as a baseline — running the broadest
    specialist even when the router thought the question was purely
    risk-flavoured costs little and guarantees the synthesizer always
    has at least one set of findings to work from.
    """
    intent = state.get("intent")
    requested: list[str] = list(intent.specialists) if intent is not None else []

    # Stable order: always start with fundamentals, then any other
    # wired specialist the router asked for. Dedupe in case the
    # router listed the same specialist twice.
    ordered: list[_RoutedSpecialist] = ["fundamentals"]
    seen = {"fundamentals"}
    for name in requested:
        if name in _WIRED_SPECIALISTS and name not in seen:
            ordered.append(name)  # type: ignore[arg-type]
            seen.add(name)
    return ordered


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
        Retrieval entry-point shared by every specialist. See
        :data:`alphamind.agents.specialists._base.RetrievalFn`.
    """
    router = make_router_node(llm, model=router_model)
    fundamentals = make_fundamentals_node(llm, retrieve, model=specialist_model)
    risk = make_risk_node(llm, retrieve, model=specialist_model)
    sentiment = make_sentiment_node(llm, retrieve, model=specialist_model)
    technical = make_technical_node(llm, retrieve, model=specialist_model)
    synthesizer = make_synthesizer_node(llm, model=synthesizer_model)
    critic = make_critic_node(llm, model=critic_model)

    graph: StateGraph[ResearchState, Any, Any, Any] = StateGraph(ResearchState)
    # LangGraph's node-callable overloads don't unify with our async-node
    # signature under strict mypy. Suppression is scoped to these seven
    # lines; nothing else in the agent layer needs the escape hatch.
    graph.add_node("router", _as_node(router))  # type: ignore[call-overload]
    graph.add_node("fundamentals", _as_node(fundamentals))  # type: ignore[call-overload]
    graph.add_node("risk", _as_node(risk))  # type: ignore[call-overload]
    graph.add_node("sentiment", _as_node(sentiment))  # type: ignore[call-overload]
    graph.add_node("technical", _as_node(technical))  # type: ignore[call-overload]
    graph.add_node("synthesizer", _as_node(synthesizer))  # type: ignore[call-overload]
    graph.add_node("critic", _as_node(critic))  # type: ignore[call-overload]

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _route_specialists,
        {
            "fundamentals": "fundamentals",
            "risk": "risk",
            "sentiment": "sentiment",
            "technical": "technical",
        },
    )
    graph.add_edge("fundamentals", "synthesizer")
    graph.add_edge("risk", "synthesizer")
    graph.add_edge("sentiment", "synthesizer")
    graph.add_edge("technical", "synthesizer")
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
