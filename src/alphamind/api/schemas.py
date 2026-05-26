"""Pydantic request/response schemas for the API layer.

Why Pydantic here when :mod:`alphamind.agents.state` deliberately
uses frozen dataclasses (ADR 0006)? Different responsibility. Agent
state needs to be transport-neutral and cheap to construct millions
of times; the API boundary needs runtime validation of untrusted
input. The two domains barely touch — the SSE serializer in
:mod:`alphamind.api.streaming` walks the dataclass state and emits
JSON; Pydantic guards the *input* side.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Body of ``POST /research``.

    The ``as_of`` field is required, not optional. Defaulting it to
    ``today`` would silently let lookahead bias into historical
    questions — the project's most important correctness invariant.
    Documented in ADR 0005 and enforced at every retrieval layer; the
    API surface enforces it too.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The research question.",
    )
    as_of: date = Field(
        ...,
        description=(
            "Time horizon (YYYY-MM-DD). No filings dated after this are "
            "used. Required to prevent lookahead bias in historical queries."
        ),
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Number of sources retrieved per specialist.",
    )


class HealthResponse(BaseModel):
    """Body of ``GET /healthz``.

    Liveness only — does not ping Postgres, Redis, or the configured
    LLM backend. A ``/readyz`` with deeper checks is out of scope
    until deployment is.
    """

    status: str = Field(default="ok")
    graph_ready: bool = Field(
        ...,
        description=(
            "True once the LangGraph DAG has been compiled on startup. "
            "Should be true whenever the process is serving traffic; a "
            "false reading means the lifespan failed and the API is up "
            "in a degraded state."
        ),
    )


__all__ = ["HealthResponse", "ResearchRequest"]
