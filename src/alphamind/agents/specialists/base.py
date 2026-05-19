"""Shared scaffolding for specialist agents.

Each specialist runs the same three-step pipeline:

1. Retrieve passages via :class:`alphamind.retrieval.search.HybridSearch`.
2. Build a source block — numbered ``[1]…[N]`` headers + raw text — and
   ask the LLM to produce a summary plus a list of cited claims.
3. Parse the JSON response into a :class:`SpecialistReport`.

The differences between specialists live in three places:

- ``name``: which slot in :class:`SpecialistReport` they fill.
- ``system_prompt``: the role and the rules of engagement.
- ``query_augmentation``: extra terms appended to the user's query
  before retrieval, so each specialist pulls a domain-relevant slice
  of the corpus instead of the same passages.

Everything else — retrieval, hydration, prompt formatting, JSON
extraction, error handling, fallback construction — lives here.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from alphamind.agents.json_utils import (
    StructuredOutputError,
    extract_json,
    require_dict,
    require_list,
    require_str,
)
from alphamind.agents.state import (
    AgentState,
    Citation,
    Claim,
    SpecialistName,
    SpecialistReport,
)
from alphamind.db.session import session_scope
from alphamind.llm.base import LLMClient, LLMClientError, SystemMessage, UserMessage
from alphamind.models.filing import Filing
from alphamind.models.filing_chunk import FilingChunk
from alphamind.retrieval.search.pipeline import HybridSearch, SearchHit

logger = logging.getLogger(__name__)


class SpecialistInvocationError(RuntimeError):
    """Raised when a specialist cannot complete its job and has no fallback."""


def build_source_block(citations: Sequence[Citation]) -> str:
    """Format citations as a numbered ``[1] header / text`` block."""
    parts: list[str] = []
    for c in citations:
        parts.append(c.header())
        parts.append(c.text)
        parts.append("")  # blank line between sources
    return "\n".join(parts).rstrip()


class SpecialistBase(ABC):
    """Common workflow for every specialist node."""

    #: Which slot the report fills. Set on the subclass.
    name: SpecialistName

    #: System prompt the LLM sees. Subclasses customise.
    system_prompt: str

    #: Extra terms appended to the user query before retrieval. Empty if
    #: the subclass doesn't want to bias retrieval.
    query_augmentation: str = ""

    def __init__(
        self,
        *,
        llm: LLMClient,
        search: HybridSearch,
        model: str | None = None,
        max_sources: int = 8,
    ) -> None:
        if max_sources <= 0:
            raise ValueError("max_sources must be positive")
        self._llm = llm
        self._search = search
        self._model = model
        self._max_sources = max_sources

    # -- LangGraph entry point ------------------------------------------------

    async def __call__(self, state: AgentState) -> dict[str, list[SpecialistReport]]:
        query = state.get("query", "")
        as_of = state.get("as_of")
        if not query.strip() or as_of is None:
            raise ValueError(f"{self.name} specialist: state.query and state.as_of are required")

        top_k = min(state.get("top_k", self._max_sources), self._max_sources)

        try:
            report = await self._run(query=query, as_of=as_of, top_k=top_k)
        except SpecialistInvocationError as exc:
            logger.warning("%s specialist failed: %s", self.name, exc)
            report = self._empty_report(reason=str(exc))

        return {"specialist_reports": [report]}

    # -- Specialist workflow --------------------------------------------------

    async def _run(self, *, query: str, as_of: date, top_k: int) -> SpecialistReport:
        retrieval_query = self._build_retrieval_query(query)

        async with session_scope() as session:
            hits = await self._search.search(
                session,
                query=retrieval_query,
                as_of=as_of,
                top_k=top_k,
            )
            if not hits:
                return self._empty_report(reason="retrieval returned no candidates")

            citations = await self._hydrate_citations(session, hits, as_of=as_of)

        if not citations:
            return self._empty_report(reason="hydration returned no citations")

        try:
            response = await self._llm.complete(
                [
                    SystemMessage(self.system_prompt),
                    UserMessage(self._build_user_prompt(query=query, citations=citations)),
                ],
                model=self._model,
                max_tokens=1200,
                temperature=0.0,
            )
        except LLMClientError as exc:
            raise SpecialistInvocationError(f"llm call failed: {exc}") from exc

        try:
            summary, claims = self._parse_response(response.content, n_sources=len(citations))
        except StructuredOutputError as exc:
            raise SpecialistInvocationError(f"could not parse specialist response: {exc}") from exc

        return SpecialistReport(
            specialist=self.name,
            summary=summary,
            claims=tuple(claims),
            citations=tuple(citations),
        )

    # -- Hooks subclasses can override ----------------------------------------

    def _build_retrieval_query(self, query: str) -> str:
        if self.query_augmentation:
            return f"{query} {self.query_augmentation}".strip()
        return query

    def _build_user_prompt(self, *, query: str, citations: Sequence[Citation]) -> str:
        sources = build_source_block(citations)
        return (
            f"Question:\n{query}\n\n"
            f"Sources (numbered 1..{len(citations)}):\n\n{sources}\n\n"
            "Respond with the JSON object specified by your system prompt."
        )

    # -- Parsing & helpers ----------------------------------------------------

    def _parse_response(
        self,
        content: str,
        *,
        n_sources: int,
    ) -> tuple[str, list[Claim]]:
        payload = require_dict(extract_json(content), context=f"{self.name}")
        summary = require_str(payload.get("summary", ""), context=f"{self.name}.summary").strip()
        raw_claims = require_list(payload.get("claims", []), context=f"{self.name}.claims")

        claims: list[Claim] = []
        for i, item in enumerate(raw_claims):
            if not isinstance(item, dict):
                logger.warning("%s: claim %d is not an object, skipping", self.name, i)
                continue
            text = require_str(
                item.get("text", ""),
                context=f"{self.name}.claims[{i}].text",
            ).strip()
            if not text:
                continue
            raw_cites = item.get("citations", [])
            citations = self._normalise_citations(raw_cites, n_sources=n_sources)
            if not citations:
                # A claim without a citation violates the rules of the
                # system prompt. Drop it rather than letting it become an
                # unsupported claim downstream.
                logger.info("%s: dropping claim without citations: %r", self.name, text)
                continue
            claims.append(Claim(text=text, citations=tuple(citations)))

        return summary, claims

    @staticmethod
    def _normalise_citations(value: Any, *, n_sources: int) -> list[int]:
        if not isinstance(value, list):
            return []
        out: list[int] = []
        seen: set[int] = set()
        for raw in value:
            try:
                ordinal = int(raw)
            except (TypeError, ValueError):
                continue
            if ordinal < 1 or ordinal > n_sources:
                continue
            if ordinal in seen:
                continue
            seen.add(ordinal)
            out.append(ordinal)
        return out

    async def _hydrate_citations(
        self,
        session: AsyncSession,
        hits: Sequence[SearchHit],
        *,
        as_of: date,
    ) -> list[Citation]:
        chunk_ids = [h.chunk_id for h in hits]
        if not chunk_ids:
            return []
        stmt = (
            select(FilingChunk)
            .where(FilingChunk.id.in_(chunk_ids))
            .where(FilingChunk.filing_date <= as_of)
            .options(selectinload(FilingChunk.filing).selectinload(Filing.company))
        )
        result = await session.execute(stmt)
        chunks = {c.id: c for c in result.scalars()}

        citations: list[Citation] = []
        for ordinal, hit in enumerate(hits, start=1):
            chunk = chunks.get(hit.chunk_id)
            if chunk is None:
                continue
            ticker = chunk.filing.company.ticker or chunk.filing.company.cik
            citations.append(
                Citation(
                    ordinal=ordinal,
                    chunk_id=chunk.id,
                    filing_id=chunk.filing_id,
                    ticker=ticker,
                    form=chunk.filing.form,
                    filing_date=chunk.filing_date,
                    section=chunk.section,
                    text=chunk.text,
                )
            )
        return citations

    def _empty_report(self, *, reason: str) -> SpecialistReport:
        return SpecialistReport(
            specialist=self.name,
            summary=f"({self.name} specialist produced no findings: {reason})",
            claims=(),
            citations=(),
        )

    # -- Subclass contract ----------------------------------------------------

    @property
    @abstractmethod
    def domain(self) -> str:
        """Human-readable label used in logging and rationale strings."""


__all__ = [
    "SpecialistBase",
    "SpecialistInvocationError",
    "build_source_block",
]
