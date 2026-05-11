"""Reranker protocol shared by every concrete implementation.

A reranker takes a query and an ordered list of candidate passages
(typically the top-N from :func:`alphamind.retrieval.hybrid_search`) and
returns the same passages sorted by a fresh per-pair relevance score.

The protocol mirrors :mod:`alphamind.embeddings.base`: a stable
``model_name`` is exposed so callers can record which reranker scored a
result, and the only required method is async-friendly so HTTP-backed
implementations remain possible later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class RerankerError(RuntimeError):
    """Raised when a reranker cannot satisfy a request."""


@dataclass(frozen=True, slots=True)
class RerankedPassage:
    """A passage's reranker score, keyed back to the input chunk."""

    chunk_id: int
    score: float


@runtime_checkable
class Reranker(Protocol):
    """A query+passage scorer.

    Implementations must be safe to call concurrently from multiple asyncio
    tasks. ``rerank`` must return one :class:`RerankedPassage` per input
    pair, sorted by ``score`` descending. Ties are broken by input order
    so the operation is deterministic when scores collide.
    """

    @property
    def model_name(self) -> str:
        """Stable identifier for this reranker (persisted with results)."""
        ...

    async def rerank(
        self,
        query: str,
        passages: list[tuple[int, str]],
    ) -> list[RerankedPassage]:
        """Score every ``(chunk_id, text)`` pair against ``query``.

        Raises :class:`RerankerError` if the backend cannot score a batch.
        """
        ...


__all__ = ["RerankedPassage", "Reranker", "RerankerError"]
