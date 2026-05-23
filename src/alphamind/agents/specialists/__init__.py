"""Specialist agents.

Each specialist is a LangGraph node that retrieves a source pool and
emits a small set of structured findings with chunk-level citations.
The synthesizer merges across specialists.

All four specialists from the architecture map are now wired:

- :func:`make_fundamentals_node` — financial-statement / MD&A material.
- :func:`make_risk_node` — Item 1A risk factors and legal proceedings.
- :func:`make_sentiment_node` — qualitative tone in MD&A and 8-K
  narrative. Earnings-transcript ingestion isn't built yet, which is
  the strongest sentiment signal; the system prompt is honest about
  the limitation.
- :func:`make_technical_node` — no-data placeholder until the
  market-data adapter exists. See its docstring for the rationale; it
  short-circuits before retrieval / LLM and emits no findings.
"""

from __future__ import annotations

from alphamind.agents.specialists.fundamentals import make_fundamentals_node
from alphamind.agents.specialists.risk import make_risk_node
from alphamind.agents.specialists.sentiment import make_sentiment_node
from alphamind.agents.specialists.technical import make_technical_node

__all__ = [
    "make_fundamentals_node",
    "make_risk_node",
    "make_sentiment_node",
    "make_technical_node",
]
