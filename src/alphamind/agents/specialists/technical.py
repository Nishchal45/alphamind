"""Technical specialist — no-data stub.

The technical specialist reasons about price action, momentum, and
volatility regime. None of those signals live in SEC filings — they
need historical OHLCV bars, corporate-action data, and a market-data
adapter that this project hasn't built yet (architecture map mentions
it; ingestion has only EDGAR today).

Rather than pretend by retrieving "stock price" mentions out of 10-Ks
(which only produces boilerplate forward-looking-statement disclaimers
and is actively misleading), this specialist ships as an honest stub:

- It is registered with the graph so the router can route to it.
- It satisfies the :class:`SpecialistBase` contract (same constructor
  signature as the others, so graph wiring doesn't need a special case).
- Its ``_run`` short-circuits and returns an empty
  :class:`SpecialistReport` with an explanatory summary.
- No LLM call, no retrieval, no cost.

When the market-data adapter lands, the override goes away and the
class falls through to the standard :meth:`SpecialistBase._run`
pipeline.
"""

from __future__ import annotations

from datetime import date

from alphamind.agents.specialists.base import SpecialistBase
from alphamind.agents.state import SpecialistReport

SYSTEM_PROMPT = """\
You are the technical analyst on an equity-research team. You reason
about price action, momentum, support / resistance, volatility regime,
and volume dynamics. (Not active yet — see class docstring.)
"""

NO_DATA_REASON = (
    "market-data adapter is not built yet; technical signals require "
    "OHLCV price / volume data not present in the SEC filing corpus"
)


class TechnicalSpecialist(SpecialistBase):
    """Stubbed specialist — returns an empty report until market data lands."""

    name = "technical"
    system_prompt = SYSTEM_PROMPT
    # Augmentation is here for the same reason ``system_prompt`` is —
    # it documents the future state. Unused while ``_run`` is overridden.
    query_augmentation = "price momentum volatility support resistance volume"

    @property
    def domain(self) -> str:
        return "technical"

    async def _run(self, *, query: str, as_of: date, top_k: int) -> SpecialistReport:
        # Short-circuit: no retrieval, no LLM call. ``query``, ``as_of``,
        # and ``top_k`` are part of the base contract and ignored here.
        del query, as_of, top_k
        return self._empty_report(reason=NO_DATA_REASON)


__all__ = ["NO_DATA_REASON", "SYSTEM_PROMPT", "TechnicalSpecialist"]
