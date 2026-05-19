"""Pydantic request / response models for the HTTP surface.

These are deliberately separate from the dataclasses in
:mod:`alphamind.agents.state`. The agent layer's dataclasses are an
internal transport optimised for immutability and Python ergonomics;
the API schemas are an external contract optimised for JSON
serialisation, validation, and documentation (FastAPI's OpenAPI
generator reads these).

Conversion happens at the boundary — :func:`event_payload_from_update`
in :mod:`alphamind.api.sse` builds these schemas from agent dataclasses
just before they go on the wire.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from alphamind.agents.state import (
    Citation,
    Claim,
    CriticReport,
    RouterDecision,
    SpecialistName,
    SpecialistReport,
    Thesis,
    UnsupportedClaim,
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
        description="Sources fed to each specialist.",
    )

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        # Trim and reject whitespace-only input that survived ``min_length=1``.
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must contain non-whitespace characters")
        return stripped


# --- SSE event payloads ------------------------------------------------------


class CitationPayload(BaseModel):
    """Mirrors :class:`Citation` for the wire."""

    ordinal: int
    chunk_id: int
    filing_id: int
    ticker: str
    form: str
    filing_date: date
    section: str | None
    text: str

    @classmethod
    def from_dataclass(cls, c: Citation) -> CitationPayload:
        return cls(
            ordinal=c.ordinal,
            chunk_id=c.chunk_id,
            filing_id=c.filing_id,
            ticker=c.ticker,
            form=c.form,
            filing_date=c.filing_date,
            section=c.section,
            text=c.text,
        )


class ClaimPayload(BaseModel):
    text: str
    citations: list[int]

    @classmethod
    def from_dataclass(cls, c: Claim) -> ClaimPayload:
        return cls(text=c.text, citations=list(c.citations))


class RouterDecisionPayload(BaseModel):
    specialists: list[SpecialistName]
    rationale: str

    @classmethod
    def from_dataclass(cls, d: RouterDecision) -> RouterDecisionPayload:
        return cls(specialists=list(d.specialists), rationale=d.rationale)


class SpecialistReportPayload(BaseModel):
    specialist: SpecialistName
    summary: str
    claims: list[ClaimPayload]
    citations: list[CitationPayload]

    @classmethod
    def from_dataclass(cls, r: SpecialistReport) -> SpecialistReportPayload:
        return cls(
            specialist=r.specialist,
            summary=r.summary,
            claims=[ClaimPayload.from_dataclass(c) for c in r.claims],
            citations=[CitationPayload.from_dataclass(c) for c in r.citations],
        )


class ThesisPayload(BaseModel):
    answer: str
    bull: list[str]
    bear: list[str]
    citations: list[CitationPayload]

    @classmethod
    def from_dataclass(cls, t: Thesis) -> ThesisPayload:
        return cls(
            answer=t.answer,
            bull=list(t.bull),
            bear=list(t.bear),
            citations=[CitationPayload.from_dataclass(c) for c in t.citations],
        )


class UnsupportedClaimPayload(BaseModel):
    claim: str
    reason: str

    @classmethod
    def from_dataclass(cls, u: UnsupportedClaim) -> UnsupportedClaimPayload:
        return cls(claim=u.claim, reason=u.reason)


class CriticReportPayload(BaseModel):
    unsupported: list[UnsupportedClaimPayload]
    notes: str
    ok: bool

    @classmethod
    def from_dataclass(cls, r: CriticReport) -> CriticReportPayload:
        return cls(
            unsupported=[UnsupportedClaimPayload.from_dataclass(u) for u in r.unsupported],
            notes=r.notes,
            ok=r.ok,
        )


class ErrorPayload(BaseModel):
    """Final SSE event when the graph itself raised."""

    message: str


# --- Health -----------------------------------------------------------------


HealthStatus = Literal["ok", "degraded", "down"]


class HealthResponse(BaseModel):
    status: HealthStatus
    detail: str | None = None


__all__ = [
    "CitationPayload",
    "ClaimPayload",
    "CriticReportPayload",
    "ErrorPayload",
    "HealthResponse",
    "HealthStatus",
    "ResearchRequest",
    "RouterDecisionPayload",
    "SpecialistReportPayload",
    "ThesisPayload",
    "UnsupportedClaimPayload",
]
