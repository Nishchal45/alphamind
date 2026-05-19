"""Typed graph state and supporting dataclasses for the agent layer.

LangGraph drives the workflow by passing a single ``AgentState`` object
through each node. The state is a ``TypedDict`` because that's what
LangGraph's runtime expects, and because partial updates (a node returns
``{"specialist_reports": [...]}`` and the runtime merges) compose more
cleanly than full-object replacements.

The leaf records (``Citation``, ``Claim``, ``SpecialistReport``, etc.)
are frozen dataclasses for the same reasons we made
:class:`alphamind.llm.base.Message` a frozen dataclass:

- Provider-agnostic transport. The synthesizer and critic don't care
  whether a claim came from a JSON LLM response or a hand-built fixture.
- Immutability. Nodes pass these around in parallel; mutating a claim
  mid-graph would be a subtle correctness bug.
- No pydantic on the hot path. Each agent run constructs dozens of
  claims; pydantic validation overhead matters once the graph fans out.

Schema validation of LLM output happens at the *boundary* (the node
that parses the response). After that, the dataclasses carry it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Literal, TypedDict

SpecialistName = Literal["fundamentals", "sentiment", "technical", "risk"]


@dataclass(frozen=True, slots=True)
class Citation:
    """One source-document reference.

    ``ordinal`` is the 1-based number the LLM uses in ``[N]`` citations
    inside its prose. ``chunk_id`` is the database key so we can audit
    back to the underlying text.
    """

    ordinal: int
    chunk_id: int
    filing_id: int
    ticker: str
    form: str
    filing_date: date
    section: str | None
    text: str

    def header(self) -> str:
        """Human-readable header used in source blocks."""
        section = self.section or "—"
        return (
            f"[{self.ordinal}] {self.ticker} — {self.form} — "
            f"{self.filing_date.isoformat()} — {section}"
        )


@dataclass(frozen=True, slots=True)
class Claim:
    """A single statement produced by a specialist, bound to its sources.

    ``citations`` are 1-based ordinals into the source block the
    specialist was shown. They are *not* chunk_ids: keeping the
    indirection lets the synthesizer renumber when it merges reports
    without touching the original prose.
    """

    text: str
    citations: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SpecialistReport:
    """A specialist's output: its citations, claims, and free-form summary."""

    specialist: SpecialistName
    summary: str
    claims: tuple[Claim, ...]
    citations: tuple[Citation, ...]


@dataclass(frozen=True, slots=True)
class Thesis:
    """The synthesizer's merged output."""

    answer: str
    bull: tuple[str, ...]
    bear: tuple[str, ...]
    # Renumbered citations across all specialists. Ordinals here are the
    # ones the critic should see.
    citations: tuple[Citation, ...]


@dataclass(frozen=True, slots=True)
class UnsupportedClaim:
    """A claim the critic could not validate against the source pool."""

    claim: str
    reason: str


@dataclass(frozen=True, slots=True)
class CriticReport:
    """Critic node output."""

    unsupported: tuple[UnsupportedClaim, ...]
    notes: str

    @property
    def ok(self) -> bool:
        return not self.unsupported


@dataclass(frozen=True, slots=True)
class RouterDecision:
    """Which specialists the router decided to run, and why."""

    specialists: tuple[SpecialistName, ...]
    rationale: str


def merge_specialist_reports(
    left: Sequence[SpecialistReport] | None,
    right: Sequence[SpecialistReport] | None,
) -> list[SpecialistReport]:
    """LangGraph reducer that concatenates specialist reports.

    Specialists run in parallel — LangGraph needs a reducer to know how
    to merge the partial updates each one returns. Concatenation
    preserves every report; deduplication isn't needed because each
    specialist runs at most once per graph invocation.
    """

    merged: list[SpecialistReport] = []
    if left:
        merged.extend(left)
    if right:
        merged.extend(right)
    return merged


class AgentState(TypedDict, total=False):
    """LangGraph state object passed between nodes.

    ``total=False`` because each node only fills in the fields it owns:

    - router writes ``router_decision``
    - each specialist appends to ``specialist_reports`` via the reducer
    - synthesizer writes ``thesis``
    - critic writes ``critic_report``

    Fields are present in the dict only after the node that owns them
    has run, which keeps the partial-update contract honest.
    """

    # --- Inputs (populated by the caller) ---
    query: str
    as_of: date
    top_k: int

    # --- Filled by router ---
    router_decision: RouterDecision

    # --- Filled by specialists (parallel-safe via concatenation reducer) ---
    specialist_reports: Annotated[list[SpecialistReport], merge_specialist_reports]

    # --- Filled by synthesizer ---
    thesis: Thesis

    # --- Filled by critic ---
    critic_report: CriticReport


ALL_SPECIALISTS: tuple[SpecialistName, ...] = (
    "fundamentals",
    "sentiment",
    "technical",
    "risk",
)


@dataclass(frozen=True, slots=True)
class GraphInput:
    """Caller-facing input wrapper.

    Mirrors :class:`AgentState` but enforces presence of the inputs at
    construction time. Use :func:`to_state` to lift it into the
    LangGraph runtime.
    """

    query: str
    as_of: date
    top_k: int = 8

    def to_state(self) -> AgentState:
        if not self.query.strip():
            raise ValueError("query must be non-empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        return AgentState(
            query=self.query,
            as_of=self.as_of,
            top_k=self.top_k,
            specialist_reports=[],
        )


__all__ = [
    "ALL_SPECIALISTS",
    "AgentState",
    "Citation",
    "Claim",
    "CriticReport",
    "GraphInput",
    "RouterDecision",
    "SpecialistName",
    "SpecialistReport",
    "Thesis",
    "UnsupportedClaim",
    "merge_specialist_reports",
]
