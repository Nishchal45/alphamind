"""Typed state for the research DAG.

LangGraph passes a single dict-like state through each node. Nodes
return partial updates that LangGraph merges into the state using a
reducer per field. Scalars are overwritten; the accumulator lists
(``sources``, ``findings``, ``usage``) are concatenated so a fan-out
across specialists doesn't clobber earlier results.

The dataclasses here describe the *content* of state: what a source
looks like, what a finding looks like, etc. They're frozen + slots to
match the rest of the codebase and to make equality-based testing cheap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from operator import add
from typing import Annotated, Literal, TypedDict

SpecialistName = Literal["fundamentals", "sentiment", "technical", "risk"]


@dataclass(frozen=True, slots=True)
class Source:
    """A retrieved chunk surfaced to the LLM.

    ``chunk_id`` is the join key the critic uses to verify that a
    citation actually exists in the pool. ``score`` is whatever the
    retriever returned (cross-encoder rerank score in production).
    """

    chunk_id: int
    filing_id: int
    ticker: str
    form: str
    filing_date: date
    section: str | None
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class Finding:
    """One supported claim emitted by a specialist.

    ``cited_chunk_ids`` must be a subset of the source pool seen by the
    specialist — the critic enforces that invariant downstream.
    """

    specialist: SpecialistName
    claim: str
    cited_chunk_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RouterIntent:
    """Router output.

    ``specialists`` is the ordered set the graph will fan out to. Today
    only ``"fundamentals"`` is wired; the router can still record richer
    intent so the synthesizer / critic can reason about coverage gaps.
    """

    primary: SpecialistName
    specialists: tuple[SpecialistName, ...]
    rationale: str


@dataclass(frozen=True, slots=True)
class ThesisClaim:
    """One bull-side or bear-side claim with its supporting chunk ids."""

    claim: str
    cited_chunk_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Thesis:
    """Synthesizer output."""

    summary: str
    bull_case: tuple[ThesisClaim, ...]
    bear_case: tuple[ThesisClaim, ...]


@dataclass(frozen=True, slots=True)
class CritiqueIssue:
    """One issue raised by the critic.

    ``cited_chunk_ids`` points back into the thesis claim under review,
    not into the source pool — it's how the caller maps an issue to the
    sentence it refers to.
    """

    kind: Literal["unsupported", "contradiction"]
    claim: str
    detail: str
    cited_chunk_ids: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Critique:
    """Critic output."""

    issues: tuple[CritiqueIssue, ...]
    parse_error: str | None = None


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage for one LLM call. Accumulated across the DAG."""

    node: str
    model: str
    input_tokens: int
    output_tokens: int


class ResearchState(TypedDict, total=False):
    """LangGraph state.

    ``total=False`` because individual nodes return partial updates;
    LangGraph merges them via the reducers declared via ``Annotated``.
    """

    # Inputs (set before the graph runs; nodes never mutate).
    query: str
    as_of: date
    top_k: int

    # Router output.
    intent: RouterIntent

    # Accumulators — concatenated across nodes via ``operator.add``.
    sources: Annotated[Sequence[Source], add]
    findings: Annotated[Sequence[Finding], add]
    usage: Annotated[Sequence[Usage], add]

    # Synthesizer + critic outputs.
    thesis: Thesis
    critique: Critique
