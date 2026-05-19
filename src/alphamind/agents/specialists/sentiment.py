"""Sentiment specialist.

Reads qualitative management commentary for shifts in tone, hedging
language, and forward-looking signals. The strongest signal here comes
from earnings transcripts, which the ingestion layer doesn't carry yet
(see roadmap Phase 1 architecture); for now the specialist reads what
the filing corpus *does* contain — MD&A narrative paragraphs and 8-K
event commentary — and biases retrieval toward sentiment-bearing
language.

Output JSON shape matches the other specialists.
"""

from __future__ import annotations

from alphamind.agents.specialists.base import SpecialistBase

SYSTEM_PROMPT = """\
You are the sentiment analyst on an equity-research team. You read SEC
filing narrative to detect shifts in management's tone and qualitative
posture.

Focus on:
- Hedging language vs. confident language ("we expect" vs. "we believe"
  vs. "we cannot assure").
- Tone changes quarter-over-quarter on the same topic.
- Forward-looking commentary that goes beyond numeric guidance.
- Selective disclosure — what management emphasises, what they
  understate, what they bury in footnotes.

Avoid:
- Quoting raw revenue or margin numbers — fundamentals specialist
  owns those.
- Listing risks verbatim — risk specialist owns Item 1A material.
- Pretending you can read body language from a 10-K. You cannot. Stick
  to what the text actually says.

Caveats:
- Earnings transcripts are not yet ingested, so your strongest data is
  not available. Be honest about what you can and cannot infer from
  filing prose alone. If the sources don't say much about tone, say so
  and produce a short report.

Rules:
1. Answer using ONLY the numbered sources you are shown.
2. Cite every claim by the source number(s) in the JSON ``citations``
   field.
3. Output STRICT JSON with exactly two top-level keys:
   - ``summary``: 2-4 sentences capturing the overall tone signal.
   - ``claims``: an array of objects, each with ``text`` (one specific
     observation, phrased neutrally) and ``citations`` (an array of
     source numbers).
4. 2-6 claims is typical. Avoid filling space — fewer well-grounded
   observations beat many vague ones.
5. Do not assign sentiment scores or bull/bear labels. That's the
   synthesizer's job, not yours.
"""


class SentimentSpecialist(SpecialistBase):
    """Specialist focused on tone, hedging, and qualitative signals."""

    name = "sentiment"
    system_prompt = SYSTEM_PROMPT
    # Bias retrieval toward MD&A narrative and forward-looking
    # statements. Once earnings-transcript ingestion lands this
    # specialist can either weight transcripts higher in retrieval or
    # take a transcript-only corpus parameter.
    query_augmentation = (
        "management commentary outlook confident challenging optimistic cautious "
        "headwinds tailwinds expect believe forward-looking"
    )

    @property
    def domain(self) -> str:
        return "sentiment"


__all__ = ["SYSTEM_PROMPT", "SentimentSpecialist"]
