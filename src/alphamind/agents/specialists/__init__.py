"""Specialist agents.

Each specialist is a LangGraph node that:

1. Runs a focused retrieval query against ``filing_chunks`` (using the
   shared :class:`alphamind.retrieval.search.HybridSearch` pipeline,
   with section / form filters that match its domain).
2. Calls the LLM with a domain-specific system prompt and the retrieved
   passages.
3. Parses the response into a :class:`SpecialistReport` with cited
   claims.

All four specialists from the architecture map ship here:

- :class:`FundamentalsSpecialist` — revenue, margins, segments, guidance.
- :class:`SentimentSpecialist` — qualitative tone, hedging, outlook
  language. Earnings-transcript corpus isn't ingested yet, so this one
  works from MD&A and 8-K narrative for now.
- :class:`RiskSpecialist` — Item 1A risk factors, legal proceedings,
  concentration risks.
- :class:`TechnicalSpecialist` — no-data stub until the market-data
  adapter exists. See its docstring for the rationale.
"""

from __future__ import annotations

from alphamind.agents.specialists.base import (
    SpecialistBase,
    SpecialistInvocationError,
    build_source_block,
)
from alphamind.agents.specialists.fundamentals import FundamentalsSpecialist
from alphamind.agents.specialists.risk import RiskSpecialist
from alphamind.agents.specialists.sentiment import SentimentSpecialist
from alphamind.agents.specialists.technical import TechnicalSpecialist

__all__ = [
    "FundamentalsSpecialist",
    "RiskSpecialist",
    "SentimentSpecialist",
    "SpecialistBase",
    "SpecialistInvocationError",
    "TechnicalSpecialist",
    "build_source_block",
]
