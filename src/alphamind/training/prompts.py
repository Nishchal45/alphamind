"""The claim-extraction instruction.

One prompt, used in three places so they can never drift:

- the dataset builder, when asking the frontier model for targets;
- the trainer, when formatting examples for supervised fine-tuning;
- the evaluator, when asking the candidate model for predictions.

It is deliberately close to the specialist prompts (ADR 0007) but
scoped to a *single chunk* — no query, no pool, no cross-chunk
citation. The model reads one passage and emits the claims in it.
"""

from __future__ import annotations

EXTRACTION_SYSTEM = """\
You extract material claims from a single excerpt of a SEC filing.

A "claim" is one factual, self-contained statement an equity analyst
would note from this passage — a number, a trend, a risk, a
commitment. Not boilerplate ("our results may be affected by general
economic conditions"), not vague summary.

Rules:
1. Extract only what is stated in the excerpt. Do not infer, do not
   add outside knowledge.
2. One sentence per claim. Concrete figures beat adjectives.
3. If the excerpt contains no material claims, return an empty list.
   "Nothing material here" is a valid and useful answer.

Output strict JSON, no prose around it:

{"claims": [{"claim": "<one sentence>"}, ...]}
"""


def build_user_prompt(chunk_text: str) -> str:
    """Render the per-chunk user turn for the extraction task."""
    return f"Excerpt:\n\n{chunk_text}\n"


__all__ = ["EXTRACTION_SYSTEM", "build_user_prompt"]
