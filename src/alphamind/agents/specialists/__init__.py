"""Specialist agents.

Each specialist is a LangGraph node that:

1. Runs a focused retrieval query against ``filing_chunks`` (using the
   shared :class:`alphamind.retrieval.search.HybridSearch` pipeline,
   with section / form filters that match its domain).
2. Calls the LLM with a domain-specific system prompt and the retrieved
   passages.
3. Parses the response into a :class:`SpecialistReport` with cited
   claims.

Only :class:`FundamentalsSpecialist` is implemented in this slice. The
other three (sentiment, technical, risk) land in follow-up PRs and
share :class:`SpecialistBase`.
"""

from __future__ import annotations

from alphamind.agents.specialists.base import (
    SpecialistBase,
    SpecialistInvocationError,
    build_source_block,
)
from alphamind.agents.specialists.fundamentals import FundamentalsSpecialist

__all__ = [
    "FundamentalsSpecialist",
    "SpecialistBase",
    "SpecialistInvocationError",
    "build_source_block",
]
