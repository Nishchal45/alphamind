"""Router node: classify intent and pick which specialists to fire.

Today only the ``fundamentals`` specialist is wired into the graph. The
router still records the broader intent so the synthesizer can flag
coverage gaps ("question implied technicals, but only fundamentals ran")
and so the graph can extend cheaply when the remaining specialists land.

If the model fails to produce parseable JSON, the router degrades to a
``fundamentals``-only intent rather than crashing the graph — losing the
critic pass over half a thesis is worse than a slightly less informed
routing decision.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal, get_args

from alphamind.agents._json import extract_json_object
from alphamind.agents.prompts import ROUTER_SYSTEM
from alphamind.agents.state import ResearchState, RouterIntent, SpecialistName, Usage
from alphamind.llm import LLMClient, UserMessage

logger = logging.getLogger(__name__)

NODE_NAME = "router"

_ALLOWED_SPECIALISTS: frozenset[SpecialistName] = frozenset(get_args(SpecialistName))


def _coerce_intent(payload: dict[str, Any]) -> RouterIntent:
    """Best-effort coercion of the model's JSON into a :class:`RouterIntent`.

    Anything unrecognised falls back to ``fundamentals``. The router is
    a hint, not a contract — we'd rather run on a sensible default than
    raise here.
    """
    raw_primary = payload.get("primary")
    primary: SpecialistName = raw_primary if raw_primary in _ALLOWED_SPECIALISTS else "fundamentals"

    raw_list = payload.get("specialists") or []
    specialists: list[SpecialistName] = [
        s for s in raw_list if isinstance(s, str) and s in _ALLOWED_SPECIALISTS
    ]
    if primary not in specialists:
        specialists.insert(0, primary)
    if not specialists:
        specialists = ["fundamentals"]

    rationale = str(payload.get("rationale") or "").strip()
    if not rationale:
        rationale = "default fundamentals routing"

    return RouterIntent(
        primary=primary,
        specialists=tuple(specialists),
        rationale=rationale,
    )


def make_router_node(
    client: LLMClient,
    *,
    model: str | None = None,
) -> Callable[[ResearchState], Awaitable[dict[str, Any]]]:
    """Return a LangGraph-compatible router node bound to ``client``."""

    async def node(state: ResearchState) -> dict[str, Any]:
        query = state["query"]
        as_of = state["as_of"]

        user = (
            f"Question: {query}\n"
            f"As-of date: {as_of.isoformat()}\n\n"
            "Return only the JSON object specified by the system prompt."
        )

        response = await client.complete(
            [UserMessage(user)],
            system=ROUTER_SYSTEM,
            model=model,
            max_tokens=512,
            temperature=0.0,
        )

        payload = extract_json_object(response.content)
        if payload is None:
            logger.warning(
                "router: failed to parse JSON, defaulting to fundamentals (got %r)",
                response.content[:200],
            )
            intent = RouterIntent(
                primary="fundamentals",
                specialists=("fundamentals",),
                rationale="parse_failure",
            )
        else:
            intent = _coerce_intent(payload)

        return {
            "intent": intent,
            "usage": [
                Usage(
                    node=NODE_NAME,
                    model=response.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
            ],
        }

    return node


_NodeLiteral = Literal["router"]
__all__ = ["NODE_NAME", "make_router_node"]
