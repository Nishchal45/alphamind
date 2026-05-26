"""Agent team — the LangGraph DAG that turns a research question into a thesis.

The graph wires seven nodes:

- :mod:`alphamind.agents.router` — classifies intent and selects specialists.
- :mod:`alphamind.agents.specialists.fundamentals` — financial-statement,
  MD&A, and business-description findings.
- :mod:`alphamind.agents.specialists.risk` — Item 1A, legal-proceedings,
  market-risk, and going-concern findings.
- :mod:`alphamind.agents.specialists.sentiment` — tone, hedging,
  forward-looking-statement signals. Filing-prose substitute until
  earnings-call transcripts are ingested.
- :mod:`alphamind.agents.specialists.technical` — quantitative trend
  signals (growth rates, margin direction, segment trajectories).
  Filing substitute until market-data ingestion lands.
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
