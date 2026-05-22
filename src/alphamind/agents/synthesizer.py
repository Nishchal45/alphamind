"""Synthesizer node.

Reads specialist findings off state and produces a structured bull /
bear thesis with chunk-level citations. The synthesizer never sees raw
filing text — only the findings the specialists distilled — which keeps
its context window small as more specialists land.

Citations are validated against the union of chunk_ids the specialists
actually cited, so the synthesizer can't invent a chunk to support a
claim. A claim with no surviving citations is dropped before the
thesis is returned.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from alphamind.agents._json import extract_json_object
from alphamind.agents.prompts import SYNTHESIZER_SYSTEM
from alphamind.agents.state import Finding, ResearchState, Thesis, ThesisClaim, Usage
from alphamind.llm import LLMClient, UserMessage

logger = logging.getLogger(__name__)

NODE_NAME = "synthesizer"


def _format_findings(findings: Sequence[Finding]) -> str:
    blocks: list[str] = []
    for i, finding in enumerate(findings, start=1):
        cites = ", ".join(str(cid) for cid in finding.cited_chunk_ids)
        blocks.append(
            f"[FINDING {i}] specialist={finding.specialist} cites=[{cites}]\n{finding.claim}"
        )
    return "\n\n".join(blocks)


def _coerce_claims(
    raw: object,
    *,
    valid_chunk_ids: set[int],
) -> tuple[ThesisClaim, ...]:
    if not isinstance(raw, list):
        return ()
    claims: list[ThesisClaim] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        if not claim:
            continue
        raw_ids = item.get("cited_chunk_ids") or []
        cited = tuple(cid for cid in raw_ids if isinstance(cid, int) and cid in valid_chunk_ids)
        if not cited:
            logger.debug("synthesizer: dropping uncited claim: %s", claim[:80])
            continue
        claims.append(ThesisClaim(claim=claim, cited_chunk_ids=cited))
    return tuple(claims)


def _empty_thesis(reason: str) -> Thesis:
    return Thesis(summary=reason, bull_case=(), bear_case=())


def make_synthesizer_node(
    client: LLMClient,
    *,
    model: str | None = None,
    max_tokens: int = 1024,
) -> Callable[[ResearchState], Awaitable[dict[str, Any]]]:
    """Return a LangGraph node bound to ``client``."""

    async def node(state: ResearchState) -> dict[str, Any]:
        findings = list(state.get("findings", []))
        if not findings:
            return {
                "thesis": _empty_thesis("No specialist findings to synthesize."),
                "usage": [],
            }

        valid_ids = {cid for f in findings for cid in f.cited_chunk_ids}
        query = state["query"]

        user = f"Question: {query}\n\nSpecialist findings:\n\n{_format_findings(findings)}\n"
        response = await client.complete(
            [UserMessage(user)],
            system=SYNTHESIZER_SYSTEM,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        payload = extract_json_object(response.content)
        if payload is None:
            logger.warning(
                "synthesizer: failed to parse JSON, emitting empty thesis (got %r)",
                response.content[:200],
            )
            thesis = _empty_thesis("Synthesizer failed to produce a parseable thesis.")
        else:
            summary = str(payload.get("summary") or "").strip()
            bull = _coerce_claims(payload.get("bull_case"), valid_chunk_ids=valid_ids)
            bear = _coerce_claims(payload.get("bear_case"), valid_chunk_ids=valid_ids)
            thesis = Thesis(
                summary=summary or "Synthesizer returned no summary.",
                bull_case=bull,
                bear_case=bear,
            )

        return {
            "thesis": thesis,
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


__all__ = ["NODE_NAME", "make_synthesizer_node"]
