"""Specialist agents.

Each specialist is a LangGraph node that retrieves a source pool and
emits a small set of structured findings with chunk-level citations.
The synthesizer merges across specialists.

The fundamentals and risk specialists are wired into the graph today;
sentiment and technical land in follow-up PRs once the upstream
ingestion is in place (earnings transcripts, market data).
"""

from __future__ import annotations

from alphamind.agents.specialists.fundamentals import make_fundamentals_node
from alphamind.agents.specialists.risk import make_risk_node

__all__ = ["make_fundamentals_node", "make_risk_node"]
