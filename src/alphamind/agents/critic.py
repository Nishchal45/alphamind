"""Critic node.

The critic is the last line of defence against hallucination. It reads
the synthesized thesis against the full source pool and emits a list
of issues:

- ``unsupported``: a thesis claim whose cited chunks do not actually
  support it (or which is missing citations entirely).
- ``contradiction``: two thesis claims that conflict, or a claim that
  contradicts what one of its cited chunks says.

We do a single machine-verifiable check ourselves before calling the
model — any thesis claim whose ``cited_chunk_ids`` is empty after
synthesizer-side validation is auto-flagged as ``unsupported``. The LLM
critique covers the harder, semantic checks.

On JSON parse failure the critic records ``parse_error`` on the
:class:`Critique` rather than crashing, so the CLI can still surface
the thesis with a clear warning that the critic couldn't run.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from alphamind.agents._json import extract_json_object
from alphamind.agents._sources import dedupe_sources
from alphamind.agents.prompts import CRITIC_SYSTEM
from alphamind.agents.state import (
    Critique,
    CritiqueIssue,
    ResearchState,
    Source,
    Thesis,
    Usage,
)
from alphamind.llm import LLMClient, UserMessage

logger = logging.getLogger(__name__)

NODE_NAME = "critic"


def _format_sources(sources: Sequence[Source]) -> str:
    blocks: list[str] = []
    for src in sources:
        section = src.section or "—"
        header = (
            f"[CHUNK {src.chunk_id}] {src.ticker} — {src.form} — "
            f"{src.filing_date.isoformat()} — {section}"
        )
        blocks.append(f"{header}\n{src.text}")
    return "\n\n".join(blocks)


def _format_thesis(thesis: Thesis) -> str:
    lines = [f"Summary: {thesis.summary}", "", "Bull case:"]
    for claim in thesis.bull_case:
        cites = ", ".join(str(c) for c in claim.cited_chunk_ids) or "(none)"
        lines.append(f"  - {claim.claim} [cites: {cites}]")
    lines.append("")
    lines.append("Bear case:")
    for claim in thesis.bear_case:
        cites = ", ".join(str(c) for c in claim.cited_chunk_ids) or "(none)"
        lines.append(f"  - {claim.claim} [cites: {cites}]")
    return "\n".join(lines)


def _coerce_issues(payload: dict[str, Any]) -> tuple[CritiqueIssue, ...]:
    raw = payload.get("issues") or []
    issues: list[CritiqueIssue] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        if kind not in ("unsupported", "contradiction"):
            continue
        claim = str(item.get("claim") or "").strip()
        detail = str(item.get("detail") or "").strip()
        if not claim or not detail:
            continue
        raw_ids = item.get("cited_chunk_ids") or []
        cited = tuple(cid for cid in raw_ids if isinstance(cid, int))
        issues.append(
            CritiqueIssue(
                kind=kind,
                claim=claim,
                detail=detail,
                cited_chunk_ids=cited,
            )
        )
    return tuple(issues)


def make_critic_node(
    client: LLMClient,
    *,
    model: str | None = None,
    max_tokens: int = 1024,
) -> Callable[[ResearchState], Awaitable[dict[str, Any]]]:
    """Return a LangGraph node bound to ``client``."""

    async def node(state: ResearchState) -> dict[str, Any]:
        thesis: Thesis | None = state.get("thesis")
        if thesis is None or (not thesis.bull_case and not thesis.bear_case):
            return {
                "critique": Critique(issues=(), parse_error=None),
                "usage": [],
            }

        sources = dedupe_sources(state.get("sources", []))
        if not sources:
            return {
                "critique": Critique(
                    issues=(
                        CritiqueIssue(
                            kind="unsupported",
                            claim="<entire thesis>",
                            detail="No source pool was supplied to the critic.",
                        ),
                    ),
                    parse_error=None,
                ),
                "usage": [],
            }

        user = (
            f"Thesis:\n\n{_format_thesis(thesis)}\n\nSource pool:\n\n{_format_sources(sources)}\n"
        )
        response = await client.complete(
            [UserMessage(user)],
            system=CRITIC_SYSTEM,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        payload = extract_json_object(response.content)
        if payload is None:
            logger.warning(
                "critic: failed to parse JSON, leaving thesis uncritiqued (got %r)",
                response.content[:200],
            )
            critique = Critique(
                issues=(),
                parse_error=(response.content[:200] or "empty response"),
            )
        else:
            critique = Critique(issues=_coerce_issues(payload), parse_error=None)

        return {
            "critique": critique,
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


__all__ = ["NODE_NAME", "make_critic_node"]
