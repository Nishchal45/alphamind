"""LangGraph DAG wiring + factory.

```
START -> router -> [selected specialists] -> synthesizer -> critic -> END
```

The router decides which specialists run; specialists that aren't
selected are short-circuited via a conditional edge so we don't pay the
LLM cost on agents whose output the synthesizer will ignore anyway.

All four specialists from the architecture map are registered:
fundamentals, sentiment, risk, and technical. The technical specialist
is a no-data stub until the market-data adapter exists; see its
class docstring for the rationale.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph

from alphamind.agents.critic import CriticNode
from alphamind.agents.router import RouterNode
from alphamind.agents.specialists import (
    FundamentalsSpecialist,
    RiskSpecialist,
    SentimentSpecialist,
    TechnicalSpecialist,
)
from alphamind.agents.specialists.base import SpecialistBase
from alphamind.agents.state import AgentState, GraphInput, SpecialistName
from alphamind.agents.synthesizer import SynthesizerNode
from alphamind.llm.base import LLMClient
from alphamind.llm.factory import get_llm_client
from alphamind.retrieval.embeddings.factory import get_embedder
from alphamind.retrieval.search.pipeline import HybridSearch
from alphamind.retrieval.search.reranker_factory import get_reranker

logger = logging.getLogger(__name__)


def _default_specialists(
    *,
    llm: LLMClient,
    search: HybridSearch,
) -> dict[SpecialistName, SpecialistBase]:
    """Return the specialists registered with the graph by default.

    Keys must match :data:`alphamind.agents.state.ALL_SPECIALISTS`.
    """

    return {
        "fundamentals": FundamentalsSpecialist(llm=llm, search=search),
        "sentiment": SentimentSpecialist(llm=llm, search=search),
        "risk": RiskSpecialist(llm=llm, search=search),
        "technical": TechnicalSpecialist(llm=llm, search=search),
    }


class ResearchGraph:
    """Compiled agent graph + an :meth:`invoke` convenience method.

    The class exists so the underlying ``CompiledStateGraph`` doesn't
    leak into call sites — callers use :meth:`invoke` with a
    :class:`GraphInput` and get an :class:`AgentState` back. Swapping
    LangGraph for something else later is a contained change.
    """

    def __init__(
        self,
        *,
        router: RouterNode,
        specialists: Mapping[SpecialistName, SpecialistBase],
        synthesizer: SynthesizerNode,
        critic: CriticNode,
    ) -> None:
        if not specialists:
            raise ValueError("at least one specialist is required")
        self._specialists = dict(specialists)
        self._compiled = self._build(
            router=router,
            specialists=self._specialists,
            synthesizer=synthesizer,
            critic=critic,
        )

    @staticmethod
    def _build(
        *,
        router: RouterNode,
        specialists: Mapping[SpecialistName, SpecialistBase],
        synthesizer: SynthesizerNode,
        critic: CriticNode,
    ) -> Any:
        builder: StateGraph[AgentState] = StateGraph(AgentState)
        builder.add_node("router", router)
        for name, node in specialists.items():
            builder.add_node(name, node)
        builder.add_node("synthesizer", synthesizer)
        builder.add_node("critic", critic)

        builder.add_edge(START, "router")

        builder.add_conditional_edges(
            "router",
            _route_to_specialists(set(specialists.keys())),
            # The mapping ensures the conditional edge's return values
            # are valid node names — including the "skip everything"
            # case which jumps straight to the synthesizer.
            {name: name for name in specialists} | {"synthesizer": "synthesizer"},
        )

        # Every specialist funnels into the synthesizer. LangGraph waits
        # for all parallel branches to complete before running the
        # downstream node.
        for name in specialists:
            builder.add_edge(name, "synthesizer")

        builder.add_edge("synthesizer", "critic")
        builder.add_edge("critic", END)
        return builder.compile()

    async def invoke(self, payload: GraphInput) -> AgentState:
        state = payload.to_state()
        # ``ainvoke`` is typed as returning the state shape but LangGraph
        # uses dynamic dict construction internally; the cast keeps the
        # public surface honest without leaking ``Any`` to callers.
        result: AgentState = await self._compiled.ainvoke(state)
        return result


def _route_to_specialists(available: set[str]) -> Any:
    """Return a LangGraph conditional-edge function.

    LangGraph's conditional edges expect the function to return either a
    single node name (single-target edge) or a list of node names
    (parallel branches). Returning a list with the names of the
    specialists the router picked fans out into a parallel branch.
    """

    def _route(state: AgentState) -> list[str] | str:
        decision = state.get("router_decision")
        if decision is None:
            chosen = available
        else:
            chosen = {s for s in decision.specialists if s in available}
        if not chosen:
            logger.info("router picked no available specialists; skipping to synthesizer")
            return "synthesizer"
        return sorted(chosen)

    return _route


def build_research_graph(
    *,
    llm: LLMClient | None = None,
    search: HybridSearch | None = None,
    specialists: Mapping[SpecialistName, SpecialistBase] | None = None,
) -> ResearchGraph:
    """Construct a :class:`ResearchGraph` for tests and library use.

    Pass your own ``llm`` / ``search`` to bypass the factories — tests
    do this with :class:`alphamind.llm.EchoLLMClient` and the
    deterministic embedder / reranker.
    """

    actual_llm = llm or get_llm_client()
    actual_search = search or HybridSearch(
        embedder=get_embedder(),
        reranker=get_reranker(),
    )
    actual_specialists = (
        specialists
        if specialists is not None
        else _default_specialists(llm=actual_llm, search=actual_search)
    )

    return ResearchGraph(
        router=RouterNode(llm=actual_llm),
        specialists=actual_specialists,
        synthesizer=SynthesizerNode(llm=actual_llm),
        critic=CriticNode(llm=actual_llm),
    )


def get_research_graph() -> ResearchGraph:
    """Process-wide research graph, using the configured LLM / search.

    Not cached: the underlying clients are cached by their own factories,
    so re-building the graph is cheap and means tests that monkey-patch
    a factory see the patched version.
    """

    return build_research_graph()


__all__ = [
    "ResearchGraph",
    "build_research_graph",
    "get_research_graph",
]
