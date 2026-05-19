"""Critic node — flags unsupported claims in the synthesizer's thesis.

The critic re-reads the thesis against the merged source pool and asks
the LLM to list claims that aren't supported. This is an LLM-judge with
structured output, not a mechanical citation-coverage check.

ADR 0007 documents the choice. The short version: a regex citation check
catches "this sentence has no [N]" but misses the failure mode that
actually matters — a citation that exists but doesn't support the claim
("Revenue grew 50% [2]" where [2] is about gross margin, not revenue).

Output JSON shape:

.. code-block:: json

    {
      "unsupported": [
        {"claim": "...", "reason": "source 2 says X, not Y"}
      ],
      "notes": "Overall the thesis is well-grounded."
    }

Fallback policy: if the critic itself fails (LLM error, malformed JSON),
the report is constructed with an empty ``unsupported`` list and a
``notes`` field that surfaces the failure. The thesis is not blocked.
The critic is a check, not a gate.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from alphamind.agents.json_utils import (
    StructuredOutputError,
    extract_json,
    require_dict,
    require_list,
    require_str,
)
from alphamind.agents.specialists.base import build_source_block
from alphamind.agents.state import (
    AgentState,
    Citation,
    CriticReport,
    Thesis,
    UnsupportedClaim,
)
from alphamind.llm.base import LLMClient, LLMClientError, SystemMessage, UserMessage

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are an adversarial reviewer auditing an equity-research thesis. Your
only job is to flag claims that the cited sources do NOT actually
support. You are not here to write the thesis or improve it.

You will receive:
- The thesis the team produced (a short answer, plus bull and bear
  bullets, all with [N] citations).
- The numbered source pool the team used.

Rules:
1. A claim is unsupported if the cited source(s) do not state the
   claim, or if the claim has no citation at all.
2. Inferences are allowed if they are well-grounded ("Revenue and gross
   margin both expanded, suggesting operating leverage" is fine when
   both numbers appear in sources).
3. Speculation, vibes, and unsupported numbers are not fine.
4. Output STRICT JSON with two top-level keys:
   - ``unsupported``: an array of objects, each with ``claim`` (the
     exact sentence or bullet, copied verbatim) and ``reason`` (one
     short sentence explaining why the cited sources do not support it).
   - ``notes``: a 1-2 sentence overall judgement.
5. If everything is well-supported, return ``"unsupported": []`` and
   say so in ``notes``.
6. Do NOT invent claims that aren't in the thesis. You are checking the
   thesis as-written, not rewriting it.
"""


def _format_thesis_for_review(thesis: Thesis) -> str:
    lines = ["## Answer", thesis.answer or "(empty)"]
    if thesis.bull:
        lines.append("")
        lines.append("## Bull case")
        for b in thesis.bull:
            lines.append(f"- {b}")
    if thesis.bear:
        lines.append("")
        lines.append("## Bear case")
        for b in thesis.bear:
            lines.append(f"- {b}")
    return "\n".join(lines)


def _build_user_prompt(*, query: str, thesis: Thesis, citations: Sequence[Citation]) -> str:
    sources = build_source_block(citations)
    body = _format_thesis_for_review(thesis)
    return (
        f"Original question:\n{query}\n\n"
        f"Thesis under review:\n\n{body}\n\n"
        f"Source pool (numbered 1..{len(citations)}):\n\n{sources}\n\n"
        "Respond with the JSON object specified by your system prompt."
    )


def _parse_report(content: str) -> CriticReport:
    payload = require_dict(extract_json(content), context="critic")
    unsupported_raw = require_list(payload.get("unsupported", []), context="critic.unsupported")
    notes = require_str(payload.get("notes", ""), context="critic.notes").strip()

    unsupported: list[UnsupportedClaim] = []
    for i, item in enumerate(unsupported_raw):
        if not isinstance(item, dict):
            logger.info("critic: skipping non-object unsupported[%d]", i)
            continue
        claim = require_str(item.get("claim", ""), context=f"critic.unsupported[{i}].claim").strip()
        reason = require_str(
            item.get("reason", ""),
            context=f"critic.unsupported[{i}].reason",
        ).strip()
        if claim:
            unsupported.append(UnsupportedClaim(claim=claim, reason=reason or "no reason provided"))

    return CriticReport(unsupported=tuple(unsupported), notes=notes)


class CriticNode:
    """LangGraph node that produces a :class:`CriticReport`."""

    def __init__(self, *, llm: LLMClient, model: str | None = None) -> None:
        self._llm = llm
        self._model = model

    async def __call__(self, state: AgentState) -> dict[str, CriticReport]:
        query = state.get("query", "")
        thesis = state.get("thesis")
        if not query.strip() or thesis is None:
            raise ValueError("critic: state.query and state.thesis are required")

        if not thesis.answer and not thesis.bull and not thesis.bear:
            return {
                "critic_report": CriticReport(
                    unsupported=(),
                    notes="empty thesis; nothing to critique",
                )
            }

        try:
            response = await self._llm.complete(
                [
                    SystemMessage(SYSTEM_PROMPT),
                    UserMessage(
                        _build_user_prompt(
                            query=query,
                            thesis=thesis,
                            citations=thesis.citations,
                        )
                    ),
                ],
                model=self._model,
                max_tokens=1500,
                temperature=0.0,
            )
            report = _parse_report(response.content)
        except (LLMClientError, StructuredOutputError) as exc:
            logger.warning("critic fallback: %s", exc)
            report = CriticReport(
                unsupported=(),
                notes=f"critic failed: {exc}",
            )

        return {"critic_report": report}


__all__ = ["SYSTEM_PROMPT", "CriticNode"]
