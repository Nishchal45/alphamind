"""Agent team — the LangGraph DAG that turns a research question into a thesis.

The graph wires four nodes:

- :mod:`alphamind.agents.router` — classifies intent and selects specialists.
- :mod:`alphamind.agents.specialists.fundamentals` — retrieves filing
  evidence and emits structured findings with chunk-level citations.
- :mod:`alphamind.agents.synthesizer` — merges specialist findings into a
  bull/bear thesis.
- :mod:`alphamind.agents.critic` — reads the thesis back against the
  source pool and flags unsupported claims and contradictions.

State is a :class:`ResearchState` TypedDict that accumulates across nodes
via LangGraph reducers; see :mod:`alphamind.agents.state`.

The graph is constructed by :func:`alphamind.agents.graph.build_research_graph`,
parameterised on an :class:`LLMClient` and a retrieval entry-point so the
DAG can be tested without a database or a real model.
"""

from __future__ import annotations

from alphamind.agents._sources import dedupe_sources
from alphamind.agents.graph import RetrievalFn, build_research_graph
from alphamind.agents.state import (
    Critique,
    Finding,
    ResearchState,
    RouterIntent,
    Source,
    Thesis,
    ThesisClaim,
    Usage,
)

__all__ = [
    "Critique",
    "Finding",
    "ResearchState",
    "RetrievalFn",
    "RouterIntent",
    "Source",
    "Thesis",
    "ThesisClaim",
    "Usage",
    "build_research_graph",
    "dedupe_sources",
]
