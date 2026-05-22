"""Fundamentals specialist node.

Pulls the top-k retrieval hits for the question + as-of, asks the LLM
for 3-7 structured findings each with chunk-id citations, and writes
both the source pool and the findings back to state.

Citations are validated post-hoc: any chunk_id the model invents that
isn't in the source pool it was shown is dropped and the finding is
kept only if at least one valid citation remains. This is cheap belt-
and-suspenders against the most common hallucination mode — the
critic still runs end-to-end across the whole thesis.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from alphamind.agents._json import extract_json_object
from alphamind.agents.prompts import FUNDAMENTALS_SYSTEM
from alphamind.agents.state import Finding, ResearchState, Source, SpecialistName, Usage
from alphamind.llm import LLMClient, UserMessage

logger = logging.getLogger(__name__)

NODE_NAME = "fundamentals"
SPECIALIST_NAME: SpecialistName = "fundamentals"


RetrievalFn = Callable[..., Awaitable[list[Source]]]
"""Pluggable retrieval entry-point.

Signature: ``async def retrieve(*, query: str, as_of: date, top_k: int)
-> list[Source]``. The graph factory takes one of these so the node
can be tested without a database — production wires the SQLAlchemy +
HybridSearch path.
"""


def _format_sources(sources: Sequence[Source]) -> str:
    """Render the source pool the way :mod:`prompts` documents."""
    blocks: list[str] = []
    for src in sources:
        section = src.section or "—"
        header = (
            f"[CHUNK {src.chunk_id}] {src.ticker} — {src.form} — "
            f"{src.filing_date.isoformat()} — {section}"
        )
        blocks.append(f"{header}\n{src.text}")
    return "\n\n".join(blocks)


def _coerce_findings(
    payload: dict[str, Any],
    *,
    valid_chunk_ids: set[int],
) -> list[Finding]:
    raw = payload.get("findings") or []
    findings: list[Finding] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        if not claim:
            continue
        raw_ids = item.get("cited_chunk_ids") or []
        cited = tuple(cid for cid in raw_ids if isinstance(cid, int) and cid in valid_chunk_ids)
        if not cited:
            logger.debug("fundamentals: dropping uncited finding: %s", claim[:80])
            continue
        findings.append(
            Finding(
                specialist=SPECIALIST_NAME,
                claim=claim,
                cited_chunk_ids=cited,
            )
        )
    return findings


def make_fundamentals_node(
    client: LLMClient,
    retrieve: RetrievalFn,
    *,
    model: str | None = None,
    max_tokens: int = 1024,
) -> Callable[[ResearchState], Awaitable[dict[str, Any]]]:
    """Return a LangGraph node bound to ``client`` and ``retrieve``."""

    async def node(state: ResearchState) -> dict[str, Any]:
        query = state["query"]
        as_of = state["as_of"]
        top_k = state.get("top_k", 8)

        sources = await retrieve(query=query, as_of=as_of, top_k=top_k)
        if not sources:
            logger.info("fundamentals: no sources retrieved; emitting no findings")
            return {"sources": [], "findings": [], "usage": []}

        user = (
            f"Question: {query}\n"
            f"As-of date: {as_of.isoformat()}\n\n"
            f"Source pool:\n\n{_format_sources(sources)}\n"
        )
        response = await client.complete(
            [UserMessage(user)],
            system=FUNDAMENTALS_SYSTEM,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        payload = extract_json_object(response.content)
        if payload is None:
            logger.warning(
                "fundamentals: failed to parse JSON, emitting no findings (got %r)",
                response.content[:200],
            )
            findings: list[Finding] = []
        else:
            findings = _coerce_findings(
                payload,
                valid_chunk_ids={src.chunk_id for src in sources},
            )

        return {
            "sources": list(sources),
            "findings": findings,
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


__all__ = ["NODE_NAME", "RetrievalFn", "make_fundamentals_node"]
