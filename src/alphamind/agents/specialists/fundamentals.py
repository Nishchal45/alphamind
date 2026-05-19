"""Fundamentals specialist.

Reads filings with an eye for revenue, margins, segments, balance-sheet
items, and forward guidance. The retrieval query is biased toward
fundamentals vocabulary so the same :class:`HybridSearch` instance
returns a different slice of the corpus than (say) the risk specialist
will when it lands.

Output JSON shape:

.. code-block:: json

    {
      "summary": "NVDA's data-center segment grew 86% YoY...",
      "claims": [
        {"text": "Data-center revenue was $14.5B in Q3.", "citations": [1]},
        {"text": "Gross margin expanded 320bps QoQ.", "citations": [2, 4]}
      ]
    }
"""

from __future__ import annotations

from alphamind.agents.specialists.base import SpecialistBase

SYSTEM_PROMPT = """\
You are the fundamentals analyst on an equity-research team. You read SEC
filings (10-K, 10-Q) and extract claims about a company's financial
performance and business model.

Focus on:
- Revenue (totals, segments, geography, growth).
- Margins (gross, operating, net) and operating leverage.
- Balance-sheet items (cash, debt, working capital).
- Guidance and forward-looking management commentary.

Avoid:
- Sentiment ("management sounded confident") — that is the sentiment
  specialist's job.
- Risk factors and litigation — risk specialist owns those.
- Price action and technicals — not your domain.

Rules:
1. Answer using ONLY the numbered sources you are shown. If the sources
   don't support a claim, do not make it.
2. Cite every claim by the source number(s) in the JSON ``citations``
   field. Numbers refer to the sources block.
3. Output STRICT JSON with exactly two top-level keys:
   - ``summary``: 2-4 sentences of plain-text overview.
   - ``claims``: an array of objects, each with ``text`` (one factual
     sentence) and ``citations`` (an array of source numbers).
4. Prefer fewer, well-supported claims over many vague ones. 3-8 claims
   is typical.
5. Do not invent numbers. If a source mentions a metric without a value,
   say so rather than fabricating one.
"""


class FundamentalsSpecialist(SpecialistBase):
    """Specialist focused on fundamentals / financial performance."""

    name = "fundamentals"
    system_prompt = SYSTEM_PROMPT
    # These terms nudge BM25 and the dense encoder toward MD&A and
    # financial-statement sections without requiring the search pipeline
    # to grow section/form filters in this PR.
    query_augmentation = "revenue margin segment guidance EPS operating cash"

    @property
    def domain(self) -> str:
        return "fundamentals"


__all__ = ["SYSTEM_PROMPT", "FundamentalsSpecialist"]
