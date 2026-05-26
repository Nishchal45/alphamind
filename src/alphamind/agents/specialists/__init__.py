"""Specialist agents.

Each specialist is a LangGraph node that retrieves a source pool and
emits a small set of structured findings with chunk-level citations.
The synthesizer merges across specialists.

All four specialists are wired into the graph today:

- ``fundamentals`` — financial-statement, MD&A, business-description
  material drawn from SEC filings.
- ``risk`` — Item 1A risk factors, legal proceedings, market-risk and
  going-concern disclosures.
- ``sentiment`` — tone, hedging, and forward-looking language. The
  natural source is earnings-call transcripts; until transcript
  ingestion lands, this specialist runs against filing prose as a
  lower-signal substitute.
- ``technical`` — quantitative trend signals (growth rates, margin
  direction, segment trajectories). Strict-sense technical analysis
  needs market-data ingestion that has not shipped yet; until it does,
  this specialist runs against filings.

Sentiment and technical share the filings retrieval surface with
fundamentals and risk today. When their natural sources land
(transcripts, market data), each specialist's retrieval entry-point
moves; the node-factory shape stays.
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
