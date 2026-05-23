"""Shared scaffolding for specialist nodes.

Every specialist follows the same shape: retrieve a source pool for
the question + as-of, call the LLM with a domain-specific system
prompt, parse JSON findings, drop hallucinated citations, write
sources + findings + usage back to state.

The factory here captures that pattern so each concrete specialist
module stays a thin wrapper around its prompt and its name.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from alphamind.agents._json import extract_json_object
from alphamind.agents.state import Finding, ResearchState, Source, SpecialistName, Usage
from alphamind.llm import LLMClient, UserMessage

logger = logging.getLogger(__name__)


RetrievalFn = Callable[..., Awaitable[list[Source]]]
"""Pluggable retrieval entry-point.

Signature: ``async def retrieve(*, query: str, as_of: date, top_k: int)
-> list[Source]``. The graph factory takes one of these so specialists
can be tested without a database — production wires the SQLAlchemy +
HybridSearch path.
"""


def _format_sources(sources: Sequence[Source]) -> str:
    """Render the source pool the way the specialist prompts document."""
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
    specialist: SpecialistName,
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
            logger.debug("%s: dropping uncited finding: %s", specialist, claim[:80])
            continue
        findings.append(
            Finding(
                specialist=specialist,
                claim=claim,
                cited_chunk_ids=cited,
            )
        )
    return findings


def make_specialist_node(
    *,
    specialist: SpecialistName,
    system_prompt: str,
    client: LLMClient,
    retrieve: RetrievalFn,
    model: str | None = None,
    max_tokens: int = 1024,
) -> Callable[[ResearchState], Awaitable[dict[str, Any]]]:
    """Return a LangGraph node for one specialist.

    Parameters
    ----------
    specialist:
        One of the SpecialistName literals; identifies the producer of
        each :class:`Finding` so the synthesizer / critic can reason
        about coverage.
    system_prompt:
        The domain-specific system prompt (e.g. FUNDAMENTALS_SYSTEM).
    client, retrieve, model, max_tokens:
        Same shape as the wider node-factory API.
    """

    async def node(state: ResearchState) -> dict[str, Any]:
        query = state["query"]
        as_of = state["as_of"]
        top_k = state.get("top_k", 8)

        sources = await retrieve(query=query, as_of=as_of, top_k=top_k)
        if not sources:
            logger.info("%s: no sources retrieved; emitting no findings", specialist)
            return {"sources": [], "findings": [], "usage": []}

        user = (
            f"Question: {query}\n"
            f"As-of date: {as_of.isoformat()}\n\n"
            f"Source pool:\n\n{_format_sources(sources)}\n"
        )
        response = await client.complete(
            [UserMessage(user)],
            system=system_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        payload = extract_json_object(response.content)
        if payload is None:
            logger.warning(
                "%s: failed to parse JSON, emitting no findings (got %r)",
                specialist,
                response.content[:200],
            )
            findings: list[Finding] = []
        else:
            findings = _coerce_findings(
                payload,
                specialist=specialist,
                valid_chunk_ids={src.chunk_id for src in sources},
            )

        return {
            "sources": list(sources),
            "findings": findings,
            "usage": [
                Usage(
                    node=specialist,
                    model=response.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
            ],
        }

    return node


__all__ = ["RetrievalFn", "make_specialist_node"]
