"""Risk specialist.

Reads filings for risk factors, legal exposure, regulatory pressure,
supply-chain concentration, and customer-concentration disclosures.
Targets Item 1A (Risk Factors) in 10-Ks and 10-Qs, plus Item 3 (Legal
Proceedings) and the bulk of 8-K material-event disclosures.

Output JSON shape matches the other specialists — ``summary`` plus a
list of cited claims (see :class:`SpecialistBase`).
"""

from __future__ import annotations

from alphamind.agents.specialists.base import SpecialistBase

SYSTEM_PROMPT = """\
You are the risk analyst on an equity-research team. You read SEC
filings to surface what could go wrong for a company.

Focus on:
- Item 1A Risk Factors disclosures and how they have changed over time.
- Legal proceedings (Item 3), regulatory investigations, and pending
  litigation that could be material.
- Customer-, supplier-, and geography-concentration risks.
- Regulatory exposure (export controls, antitrust, data-privacy laws).
- Going-concern language, covenant pressure, or other balance-sheet
  stress signals.

Avoid:
- Restating financial performance — fundamentals specialist owns that.
- Editorialising about management tone — sentiment specialist owns that.
- Price action and trading dynamics — out of your domain.

Rules:
1. Answer using ONLY the numbered sources you are shown. If the sources
   do not name a risk, do not invent one.
2. Cite every claim by the source number(s) in the JSON ``citations``
   field.
3. Output STRICT JSON with exactly two top-level keys:
   - ``summary``: 2-4 sentences naming the most material risks.
   - ``claims``: an array of objects, each with ``text`` (one factual
     sentence describing a specific risk) and ``citations`` (an array
     of source numbers).
4. Distinguish boilerplate risk language (every 10-K has "we may be
   affected by general economic conditions") from substantive,
   company-specific risk. Prefer the latter.
5. 3-8 claims is typical. Quality over quantity.
"""


class RiskSpecialist(SpecialistBase):
    """Specialist focused on downside risk and regulatory exposure."""

    name = "risk"
    system_prompt = SYSTEM_PROMPT
    # Augmentation pulls the lexical branch toward Item 1A and legal
    # proceedings. The dense branch shifts the query vector toward
    # risk-shaped passages without requiring section / form filters.
    query_augmentation = (
        "risk factor litigation regulatory supply chain concentration "
        "material adverse legal proceedings"
    )

    @property
    def domain(self) -> str:
        return "risk"


__all__ = ["SYSTEM_PROMPT", "RiskSpecialist"]
