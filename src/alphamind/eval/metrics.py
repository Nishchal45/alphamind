"""Pure metric functions over a finished :class:`ResearchState`.

Every function returns a float in ``[0, 1]`` or ``None`` (when the
case did not supply the hint the metric depends on). They are
deliberately stateless and side-effect free so they can be tested in
isolation and so the runner can call them without ordering concerns.

Definitions
-----------
- **citation_coverage**: fraction of thesis claims (bull + bear) that
  carry at least one chunk-id citation. Bound to ``[0, 1]``; ``1.0``
  when the thesis is empty (vacuously: no uncited claims).
- **citation_validity**: fraction of cited chunk ids that actually
  appear in the source pool. Should always be ``1.0`` in practice
  because the specialist and synthesizer drop invalid cites before
  the thesis leaves them — this metric is the regression alarm.
- **hallucination_rate**: fraction of thesis claims the critic
  flagged as ``unsupported``. Defined as
  ``unsupported_issues / total_claims``; ``0.0`` when the thesis is
  empty.
- **contradiction_rate**: fraction of thesis claims the critic
  flagged as ``contradiction``. Same denominator.
- **topic_recall**: fraction of the case's ``expected_topics``
  keywords (case-insensitive substring match) that land in the
  thesis text (summary + bull + bear). Returns ``None`` when the
  case doesn't supply expected_topics.
- **chunk_recall**: fraction of the case's ``required_chunk_ids``
  that appear in the cited set of any thesis claim. Returns ``None``
  when the case doesn't supply required_chunk_ids.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from alphamind.agents.state import Critique, ResearchState, Source, Thesis, ThesisClaim


def _all_claims(thesis: Thesis | None) -> list[ThesisClaim]:
    if thesis is None:
        return []
    return list(thesis.bull_case) + list(thesis.bear_case)


def _thesis_text(thesis: Thesis | None) -> str:
    if thesis is None:
        return ""
    parts: list[str] = [thesis.summary]
    parts.extend(c.claim for c in thesis.bull_case)
    parts.extend(c.claim for c in thesis.bear_case)
    return " ".join(parts)


def citation_coverage(state: ResearchState) -> float:
    """Fraction of thesis claims with at least one citation."""
    claims = _all_claims(state.get("thesis"))
    if not claims:
        return 1.0
    cited = sum(1 for c in claims if c.cited_chunk_ids)
    return cited / len(claims)


def citation_validity(state: ResearchState) -> float:
    """Fraction of cited chunk ids that exist in the source pool."""
    claims = _all_claims(state.get("thesis"))
    sources: Sequence[Source] = state.get("sources", [])
    valid_ids = {s.chunk_id for s in sources}

    total = 0
    valid = 0
    for claim in claims:
        for cid in claim.cited_chunk_ids:
            total += 1
            if cid in valid_ids:
                valid += 1
    if total == 0:
        return 1.0
    return valid / total


def _critic_issue_rate(state: ResearchState, kind: str) -> float:
    claims = _all_claims(state.get("thesis"))
    if not claims:
        return 0.0
    critique: Critique | None = state.get("critique")
    if critique is None:
        return 0.0
    flagged = sum(1 for issue in critique.issues if issue.kind == kind)
    return flagged / len(claims)


def hallucination_rate(state: ResearchState) -> float:
    """Fraction of thesis claims the critic flagged as unsupported."""
    return _critic_issue_rate(state, "unsupported")


def contradiction_rate(state: ResearchState) -> float:
    """Fraction of thesis claims the critic flagged as contradiction."""
    return _critic_issue_rate(state, "contradiction")


def topic_recall(
    state: ResearchState,
    expected_topics: Iterable[str],
) -> float | None:
    """Fraction of expected-topic keywords present in the thesis text.

    Case-insensitive substring match. Returns ``None`` when the case
    didn't supply any expected topics — the runner reports those as
    "not measured" rather than averaging zeros into the aggregate.
    """
    topics = [t for t in expected_topics if t.strip()]
    if not topics:
        return None
    text = _thesis_text(state.get("thesis")).lower()
    if not text:
        return 0.0
    hit = sum(1 for t in topics if t.lower() in text)
    return hit / len(topics)


def chunk_recall(
    state: ResearchState,
    required_chunk_ids: Iterable[int],
) -> float | None:
    """Fraction of required chunk ids that landed in some thesis claim."""
    required = list(required_chunk_ids)
    if not required:
        return None
    cited: set[int] = set()
    for claim in _all_claims(state.get("thesis")):
        cited.update(claim.cited_chunk_ids)
    hit = sum(1 for cid in required if cid in cited)
    return hit / len(required)


__all__ = [
    "chunk_recall",
    "citation_coverage",
    "citation_validity",
    "contradiction_rate",
    "hallucination_rate",
    "topic_recall",
]
