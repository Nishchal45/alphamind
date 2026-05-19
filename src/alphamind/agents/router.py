"""Router node — picks which specialists to run for a given query.

The router takes one LLM call. It's cheap because routing decisions
don't need a frontier model, but we go through the same
:class:`alphamind.llm.base.LLMClient` protocol as everything else so the
caller can downshift to a small model in a follow-up without changing
this code.

Output shape (JSON):

.. code-block:: json

    {
      "specialists": ["fundamentals", "risk"],
      "rationale": "Question is about revenue concentration..."
    }

Why structured JSON instead of tool use:

- Tool use isn't in the :class:`LLMClient` protocol yet (ADR 0006 calls
  this out — it lands when the next agent slice needs it).
- A JSON object is enough to express "which specialists" and a short
  rationale. Anything richer is over-engineering for v1.

Fallback policy: if the LLM response can't be parsed, or the model
selects no specialists, the router falls back to running every
specialist. That's safer than refusing to answer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from alphamind.agents.json_utils import (
    StructuredOutputError,
    extract_json,
    require_dict,
    require_list,
    require_str,
)
from alphamind.agents.state import (
    ALL_SPECIALISTS,
    AgentState,
    RouterDecision,
    SpecialistName,
)
from alphamind.llm.base import LLMClient, LLMClientError, SystemMessage, UserMessage

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are the router for an equity-research multi-agent system. Your job is
to pick which specialist agents should work on a research question.

Available specialists:
- fundamentals: revenue, margins, segments, balance-sheet items, guidance.
  Sources: 10-K / 10-Q MD&A, financial statements.
- sentiment: tone of management commentary, analyst-call dynamics,
  qualitative shifts quarter-over-quarter.
- technical: price action, momentum, volatility regime. Not used unless
  the question is explicitly about price or trading setups.
- risk: 1A risk factors, legal proceedings, regulatory exposure,
  supply-chain concentration.

Rules:
1. Return STRICT JSON with two keys: "specialists" (an array of names
   from the list above) and "rationale" (one short sentence).
2. Pick the smallest set that covers the question. Two specialists is
   typical; four is rare. Empty arrays are not allowed.
3. Do not invent specialists outside the list. If unsure, include
   "fundamentals" as a default.
"""


def _build_prompt(query: str) -> str:
    return f"Question:\n{query}\n\nReturn the JSON object."


def _validate_specialists(raw: Sequence[Any]) -> tuple[SpecialistName, ...]:
    valid: list[SpecialistName] = []
    seen: set[SpecialistName] = set()
    allowed: set[str] = set(ALL_SPECIALISTS)
    for item in raw:
        if not isinstance(item, str):
            continue
        name = item.strip().lower()
        if name not in allowed:
            continue
        # mypy: narrowed above via the allowed-set check.
        narrowed: SpecialistName = name  # type: ignore[assignment]
        if narrowed in seen:
            continue
        seen.add(narrowed)
        valid.append(narrowed)
    return tuple(valid)


def _parse_decision(content: str) -> RouterDecision:
    payload = require_dict(extract_json(content), context="router")
    specialists_raw = require_list(payload.get("specialists", []), context="router.specialists")
    specialists = _validate_specialists(specialists_raw)
    rationale = require_str(payload.get("rationale", ""), context="router.rationale").strip()
    if not specialists:
        raise StructuredOutputError("router returned no valid specialists")
    return RouterDecision(specialists=specialists, rationale=rationale)


class RouterNode:
    """LangGraph node that decides which specialists to run."""

    def __init__(self, *, llm: LLMClient, model: str | None = None) -> None:
        self._llm = llm
        self._model = model

    async def __call__(self, state: AgentState) -> dict[str, RouterDecision]:
        query = state.get("query", "")
        if not query.strip():
            raise ValueError("router: state.query must be set")

        try:
            response = await self._llm.complete(
                [SystemMessage(SYSTEM_PROMPT), UserMessage(_build_prompt(query))],
                model=self._model,
                max_tokens=300,
                temperature=0.0,
            )
            decision = _parse_decision(response.content)
        except (LLMClientError, StructuredOutputError) as exc:
            logger.warning(
                "router fallback to all-specialists: %s",
                exc,
            )
            decision = RouterDecision(
                specialists=ALL_SPECIALISTS,
                rationale=f"fallback after router error: {exc}",
            )

        logger.info(
            "router picked specialists=%s rationale=%r",
            decision.specialists,
            decision.rationale,
        )
        return {"router_decision": decision}


__all__ = ["SYSTEM_PROMPT", "RouterNode"]
