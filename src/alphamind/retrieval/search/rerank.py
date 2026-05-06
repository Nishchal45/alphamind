"""Cross-encoder reranker — protocol + deterministic stub.

The protocol is what the search pipeline depends on. Concrete reranker
implementations:

- :class:`DeterministicReranker` — token-overlap based. Tests the wiring;
  it's not a good reranker. Real semantic reranking comes from a cross-
  encoder (e.g., ``cross-encoder/ms-marco-MiniLM-L-12-v2``) wired in
  Phase 3.

The stub exists for the same reason as the deterministic embedder: lets
the full pipeline run end-to-end without a model download.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class RerankerError(RuntimeError):
    """Raised when the reranker cannot score an input."""


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    """A candidate passed into the reranker."""

    chunk_id: int
    text: str


@dataclass(frozen=True, slots=True)
class RerankedHit:
    """A reranker's output for one candidate."""

    chunk_id: int
    score: float


@runtime_checkable
class Reranker(Protocol):
    """Scores ``(query, candidate)`` pairs and returns descending-score hits."""

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> list[RerankedHit]: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class DeterministicReranker:
    """Jaccard-overlap reranker. Useless for production, fine for tests.

    Score is the intersection-over-union of tokenised query and candidate.
    Two candidates with identical text relative to the query get identical
    scores; ordering is stable in the input order for ties.
    """

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> list[RerankedHit]:
        if not candidates:
            return []

        q_tokens = _tokens(query)
        scored: list[tuple[int, RerankedHit]] = []
        for idx, cand in enumerate(candidates):
            c_tokens = _tokens(cand.text)
            if not q_tokens and not c_tokens:
                score = 0.0
            else:
                union = len(q_tokens | c_tokens) or 1
                score = len(q_tokens & c_tokens) / union
            scored.append((idx, RerankedHit(chunk_id=cand.chunk_id, score=score)))

        # Sort by score descending, original index ascending for stability.
        scored.sort(key=lambda pair: (-pair[1].score, pair[0]))
        return [hit for _, hit in scored]
