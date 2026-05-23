"""Pydantic request / response models for the HTTP surface.

Deliberately separate from the dataclasses in
:mod:`alphamind.agents.state`. The agent layer's records are an
internal transport optimised for immutability and ergonomic Python;
the API schemas are an external contract optimised for JSON
serialisation, validation, and OpenAPI doc generation.

Conversion happens at the boundary — :mod:`alphamind.api.sse` builds
these schemas from agent dataclasses just before they go on the wire.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from alphamind.agents.state import (
    CritiqueIssue,
    Finding,
    RouterIntent,
    Source,
    SpecialistName,
    Thesis,
    ThesisClaim,
    Usage,
)


class ResearchRequest(BaseModel):
    """POST /research body."""

    query: str = Field(min_length=1, description="The research question.")
    as_of: date = Field(
        description=(
            "Time horizon (YYYY-MM-DD). No filings dated after this are "
            "used. Required, never defaulted — see ADR 0005."
        ),
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Sources retrieved per specialist.",
    )

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must contain non-whitespace characters")
        return stripped


# --- SSE event payloads ------------------------------------------------------


class SourcePayload(BaseModel):
    """Mirror of :class:`Source` for the wire."""

    chunk_id: int
    filing_id: int
    ticker: str
    form: str
    filing_date: date
    section: str | None
    text: str
    score: float

    @classmethod
    def from_dataclass(cls, s: Source) -> SourcePayload:
        return cls(
            chunk_id=s.chunk_id,
            filing_id=s.filing_id,
            ticker=s.ticker,
            form=s.form,
            filing_date=s.filing_date,
            section=s.section,
            text=s.text,
            score=s.score,
        )


class FindingPayload(BaseModel):
    specialist: SpecialistName
    claim: str
    cited_chunk_ids: list[int]

    @classmethod
    def from_dataclass(cls, f: Finding) -> FindingPayload:
        return cls(
            specialist=f.specialist,
            claim=f.claim,
            cited_chunk_ids=list(f.cited_chunk_ids),
        )


class UsagePayload(BaseModel):
    node: str
    model: str
    input_tokens: int
    output_tokens: int

    @classmethod
    def from_dataclass(cls, u: Usage) -> UsagePayload:
        return cls(
            node=u.node,
            model=u.model,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
        )


class RouterIntentPayload(BaseModel):
    primary: SpecialistName
    specialists: list[SpecialistName]
    rationale: str

    @classmethod
    def from_dataclass(cls, i: RouterIntent) -> RouterIntentPayload:
        return cls(
            primary=i.primary,
            specialists=list(i.specialists),
            rationale=i.rationale,
        )


class ThesisClaimPayload(BaseModel):
    claim: str
    cited_chunk_ids: list[int]

    @classmethod
    def from_dataclass(cls, c: ThesisClaim) -> ThesisClaimPayload:
        return cls(claim=c.claim, cited_chunk_ids=list(c.cited_chunk_ids))


class ThesisPayload(BaseModel):
    summary: str
    bull_case: list[ThesisClaimPayload]
    bear_case: list[ThesisClaimPayload]

    @classmethod
    def from_dataclass(cls, t: Thesis) -> ThesisPayload:
        return cls(
            summary=t.summary,
            bull_case=[ThesisClaimPayload.from_dataclass(c) for c in t.bull_case],
            bear_case=[ThesisClaimPayload.from_dataclass(c) for c in t.bear_case],
        )


class CritiqueIssuePayload(BaseModel):
    kind: Literal["unsupported", "contradiction"]
    claim: str
    detail: str
    cited_chunk_ids: list[int]

    @classmethod
    def from_dataclass(cls, i: CritiqueIssue) -> CritiqueIssuePayload:
        return cls(
            kind=i.kind,
            claim=i.claim,
            detail=i.detail,
            cited_chunk_ids=list(i.cited_chunk_ids),
        )


class CritiquePayload(BaseModel):
    """Critic output bundled for one SSE event.

    ``parse_error`` is non-null when the critic LLM emitted unparseable
    output. The thesis isn't blocked — the critic is a check, not a
    gate — but the client should surface the failure rather than
    pretending "no issues" means "clean."
    """

    issues: list[CritiqueIssuePayload]
    parse_error: str | None

    @classmethod
    def from_dataclass_with_findings(
        cls,
        critique_issues: list[CritiqueIssuePayload],
        parse_error: str | None,
    ) -> CritiquePayload:
        return cls(issues=critique_issues, parse_error=parse_error)


class SpecialistEventPayload(BaseModel):
    """Per-specialist node update: which specialist ran, and what it found."""

    specialist: SpecialistName
    sources: list[SourcePayload]
    findings: list[FindingPayload]
    usage: list[UsagePayload]


class ErrorPayload(BaseModel):
    """Final SSE event when the graph itself raised."""

    message: str


# --- Health -----------------------------------------------------------------


HealthStatus = Literal["ok", "degraded", "down"]


class HealthResponse(BaseModel):
    status: HealthStatus
    detail: str | None = None


__all__ = [
    "CritiqueIssuePayload",
    "CritiquePayload",
    "ErrorPayload",
    "FindingPayload",
    "HealthResponse",
    "HealthStatus",
    "ResearchRequest",
    "RouterIntentPayload",
    "SourcePayload",
    "SpecialistEventPayload",
    "ThesisClaimPayload",
    "ThesisPayload",
    "UsagePayload",
]
