"""Agent layer for AlphaMind.

Phase 3 of the project. A small LangGraph DAG turns the retrieval pipeline
into a research workflow:

```
START -> router -> specialists -> synthesizer -> critic -> END
```

- :mod:`alphamind.agents.state` — typed graph state, citations, claims,
  specialist reports, and thesis dataclasses.
- :mod:`alphamind.agents.router` — intent classification + specialist
  selection via structured LLM output.
- :mod:`alphamind.agents.specialists` — domain-specific agents that
  retrieve from ``filing_chunks`` and emit cited claims.
- :mod:`alphamind.agents.synthesizer` — merges specialist reports into a
  structured thesis (bull / bear / answer).
- :mod:`alphamind.agents.critic` — LLM-judge that flags unsupported
  claims by re-reading the source pool.
- :mod:`alphamind.agents.graph` — wires the DAG and exposes
  :func:`get_research_graph`.

Every node depends on :class:`alphamind.llm.base.LLMClient` (not the
Anthropic SDK), so the graph runs end-to-end with ``LLM_BACKEND=echo``
for offline development and tests — same contract as the rest of the
project.
"""

from __future__ import annotations

from alphamind.agents.graph import (
    ResearchGraph,
    build_research_graph,
    get_research_graph,
)
from alphamind.agents.state import (
    AgentState,
    Citation,
    Claim,
    CriticReport,
    GraphInput,
    RouterDecision,
    SpecialistName,
    SpecialistReport,
    Thesis,
    UnsupportedClaim,
)

__all__ = [
    "AgentState",
    "Citation",
    "Claim",
    "CriticReport",
    "GraphInput",
    "ResearchGraph",
    "RouterDecision",
    "SpecialistName",
    "SpecialistReport",
    "Thesis",
    "UnsupportedClaim",
    "build_research_graph",
    "get_research_graph",
]
