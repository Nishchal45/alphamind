"""Synthesizer node — merges specialist reports into a structured thesis.

Inputs: one or more :class:`SpecialistReport` objects sitting in
``state["specialist_reports"]``. Each report has its own 1..N citation
numbering local to that specialist.

Output: a :class:`Thesis` with:

- ``answer`` — a 2-4 sentence direct answer to the query.
- ``bull`` — bullet points supporting the constructive case.
- ``bear`` — bullet points supporting the cautious case.
- ``citations`` — a single re-numbered list of citations spanning all
  specialists. Ordinals in the prose refer to this merged list.

The renumbering matters: if the fundamentals specialist cited ``[1]``
meaning *its* first source, and the risk specialist also cited ``[1]``
meaning *its* first source, the synthesizer can't pass both through to
the critic with overlapping numbers. The merge happens here, before the
LLM sees anything.
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
    SpecialistReport,
    Thesis,
)
from alphamind.llm.base import LLMClient, LLMClientError, SystemMessage, UserMessage

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are the synthesizer for an equity-research team. Several specialists
have produced reports with their own findings. Merge them into one
structured thesis.

You will receive:
- The original research question.
- Per-specialist summaries, in plain text.
- A single merged list of numbered sources [1]..[N]. All citations in
  your output must refer to this merged list.

Rules:
1. Answer using ONLY the merged sources. Cite every concrete claim with
   the source number(s) in square brackets: e.g. "Revenue grew 23% [3]."
2. Produce STRICT JSON with exactly four top-level keys:
   - ``answer``: 2-4 sentences answering the question.
   - ``bull``: an array of 2-5 short bullet points (each a single
     sentence) for the constructive case.
   - ``bear``: an array of 2-5 short bullet points for the cautious case.
3. Every bullet and every sentence in ``answer`` must contain at least
   one ``[N]`` citation. Bullets without citations will be flagged by
   the critic.
4. If specialists contradict each other, surface the disagreement
   honestly. Do not paper over it.
5. Do not introduce facts that are not in the sources.
"""


def _merge_citations(reports: Sequence[SpecialistReport]) -> list[Citation]:
    """Concatenate citations from each report into one renumbered list.

    Within one specialist, ordinal X always refers to the same chunk_id.
    Across specialists, the same chunk may appear under different
    ordinals (e.g. ``[2]`` in fundamentals and ``[1]`` in risk). We
    deduplicate by ``chunk_id`` so the synthesizer doesn't see the same
    source twice.
    """

    merged: list[Citation] = []
    seen: set[int] = set()
    next_ordinal = 1
    for report in reports:
        for citation in report.citations:
            if citation.chunk_id in seen:
                continue
            seen.add(citation.chunk_id)
            merged.append(
                Citation(
                    ordinal=next_ordinal,
                    chunk_id=citation.chunk_id,
                    filing_id=citation.filing_id,
                    ticker=citation.ticker,
                    form=citation.form,
                    filing_date=citation.filing_date,
                    section=citation.section,
                    text=citation.text,
                )
            )
            next_ordinal += 1
    return merged


def _format_specialist_summaries(reports: Sequence[SpecialistReport]) -> str:
    parts: list[str] = []
    for report in reports:
        parts.append(f"## {report.specialist.upper()}")
        parts.append(report.summary or "(no summary)")
        if report.claims:
            parts.append("")
            parts.append("Claims:")
            for claim in report.claims:
                cite_str = (
                    "[" + "][".join(str(c) for c in claim.citations) + "]"
                    if claim.citations
                    else ""
                )
                parts.append(f"- {claim.text} {cite_str}".rstrip())
        parts.append("")
    return "\n".join(parts).rstrip()


def _build_user_prompt(
    *,
    query: str,
    reports: Sequence[SpecialistReport],
    merged: Sequence[Citation],
) -> str:
    summaries = _format_specialist_summaries(reports)
    sources = build_source_block(merged)
    return (
        f"Question:\n{query}\n\n"
        f"Specialist reports:\n\n{summaries}\n\n"
        f"Merged sources (numbered 1..{len(merged)}):\n\n{sources}\n\n"
        "Note: the specialist reports above used local citation numbers. "
        "Translate them into the merged source numbers when you write "
        "the thesis. Respond with the JSON object specified by your "
        "system prompt."
    )


def _parse_thesis(content: str, *, citations: list[Citation]) -> Thesis:
    payload = require_dict(extract_json(content), context="synthesizer")
    answer = require_str(payload.get("answer", ""), context="synthesizer.answer").strip()
    bull_raw = require_list(payload.get("bull", []), context="synthesizer.bull")
    bear_raw = require_list(payload.get("bear", []), context="synthesizer.bear")

    bull = tuple(
        require_str(item, context=f"synthesizer.bull[{i}]").strip()
        for i, item in enumerate(bull_raw)
        if isinstance(item, str) and item.strip()
    )
    bear = tuple(
        require_str(item, context=f"synthesizer.bear[{i}]").strip()
        for i, item in enumerate(bear_raw)
        if isinstance(item, str) and item.strip()
    )

    return Thesis(
        answer=answer,
        bull=bull,
        bear=bear,
        citations=tuple(citations),
    )


class SynthesizerNode:
    """LangGraph node that merges specialist reports into a thesis."""

    def __init__(self, *, llm: LLMClient, model: str | None = None) -> None:
        self._llm = llm
        self._model = model

    async def __call__(self, state: AgentState) -> dict[str, Thesis]:
        query = state.get("query", "")
        reports: list[SpecialistReport] = list(state.get("specialist_reports", []))
        if not query.strip():
            raise ValueError("synthesizer: state.query must be set")

        citations = _merge_citations(reports)

        if not reports or not citations:
            return {
                "thesis": Thesis(
                    answer="No specialist reports were produced; cannot synthesize a thesis.",
                    bull=(),
                    bear=(),
                    citations=tuple(citations),
                )
            }

        try:
            response = await self._llm.complete(
                [
                    SystemMessage(SYSTEM_PROMPT),
                    UserMessage(_build_user_prompt(query=query, reports=reports, merged=citations)),
                ],
                model=self._model,
                max_tokens=1500,
                temperature=0.0,
            )
            thesis = _parse_thesis(response.content, citations=citations)
        except (LLMClientError, StructuredOutputError) as exc:
            logger.warning("synthesizer fallback: %s", exc)
            thesis = Thesis(
                answer=f"(synthesizer failed: {exc})",
                bull=(),
                bear=(),
                citations=tuple(citations),
            )

        return {"thesis": thesis}


__all__ = ["SYSTEM_PROMPT", "SynthesizerNode"]
