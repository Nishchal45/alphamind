"""Smoke test for the hybrid search pipeline orchestration.

Uses monkeypatch to swap out the two DB-dependent helpers (``lexical_search``
and ``dense_search``) and a fake AsyncSession that satisfies ``_hydrate``.
The point is to verify the orchestration glue, not the SQL — those go in
integration tests against a real Postgres instance.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import pytest

from alphamind.retrieval.embeddings.deterministic import DeterministicHashEmbedder
from alphamind.retrieval.search import pipeline as pipeline_module
from alphamind.retrieval.search.dense import DenseHit
from alphamind.retrieval.search.lexical import LexicalHit
from alphamind.retrieval.search.pipeline import HybridSearch
from alphamind.retrieval.search.rerank import DeterministicReranker

pytestmark = pytest.mark.asyncio


class _StubChunk:
    """Minimal duck-typed FilingChunk for hydration."""

    def __init__(
        self,
        *,
        chunk_id: int,
        filing_id: int,
        filing_date: date,
        section: str | None,
        text: str,
    ) -> None:
        self.id = chunk_id
        self.filing_id = filing_id
        self.filing_date = filing_date
        self.section = section
        self.text = text


class _StubResult:
    def __init__(self, rows: Sequence[Any]) -> None:
        self._rows = list(rows)

    def scalars(self) -> Sequence[Any]:
        return self._rows


class _FakeSession:
    """Returns canned hydration results from a chunk store."""

    def __init__(self, chunks: list[_StubChunk]) -> None:
        self._chunks = {c.id: c for c in chunks}
        self.executed: list[Any] = []

    async def execute(self, stmt: Any) -> _StubResult:
        self.executed.append(stmt)
        return _StubResult(list(self._chunks.values()))


async def test_pipeline_orders_results_by_rerank_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        _StubChunk(
            chunk_id=1,
            filing_id=10,
            filing_date=date(2024, 6, 1),
            section="Item 1A. Risk Factors",
            text="Inventory write-down details. Apple recorded a charge.",
        ),
        _StubChunk(
            chunk_id=2,
            filing_id=10,
            filing_date=date(2024, 6, 1),
            section="Item 7. MD&A",
            text="Sales rose year over year. No inventory issue mentioned.",
        ),
        _StubChunk(
            chunk_id=3,
            filing_id=11,
            filing_date=date(2024, 1, 1),
            section=None,
            text="Boilerplate forward-looking statement disclaimer.",
        ),
    ]

    async def fake_lexical(*args: Any, **kwargs: Any) -> list[LexicalHit]:
        return [
            LexicalHit(chunk_id=1, score=0.9),
            LexicalHit(chunk_id=2, score=0.5),
            LexicalHit(chunk_id=3, score=0.1),
        ]

    async def fake_dense(*args: Any, **kwargs: Any) -> list[DenseHit]:
        return [
            DenseHit(chunk_id=2, score=0.95),
            DenseHit(chunk_id=1, score=0.92),
            DenseHit(chunk_id=3, score=0.40),
        ]

    monkeypatch.setattr(pipeline_module, "lexical_search", fake_lexical)
    monkeypatch.setattr(pipeline_module, "dense_search", fake_dense)

    search = HybridSearch(
        embedder=DeterministicHashEmbedder(),
        reranker=DeterministicReranker(),
        candidate_pool_size=10,
        rerank_pool_size=5,
    )
    session = _FakeSession(chunks)

    hits = await search.search(
        session,  # type: ignore[arg-type]
        query="inventory write-down",
        as_of=date(2025, 1, 1),
        top_k=3,
    )

    assert len(hits) == 3
    # Chunk 1 is most lexically aligned with "inventory write-down" — the
    # deterministic reranker should put it first.
    assert hits[0].chunk_id == 1
    # Boilerplate disclaimer (chunk 3) should rank last.
    assert hits[-1].chunk_id == 3


async def test_empty_query_returns_no_hits() -> None:
    search = HybridSearch(
        embedder=DeterministicHashEmbedder(),
        reranker=DeterministicReranker(),
    )
    session = _FakeSession([])

    hits = await search.search(
        session,  # type: ignore[arg-type]
        query="   ",
        as_of=date(2025, 1, 1),
    )

    assert hits == []


async def test_construction_validates_pool_sizes() -> None:
    embedder = DeterministicHashEmbedder()
    reranker = DeterministicReranker()

    with pytest.raises(ValueError):
        HybridSearch(embedder=embedder, reranker=reranker, candidate_pool_size=0)
    with pytest.raises(ValueError):
        HybridSearch(embedder=embedder, reranker=reranker, rerank_pool_size=0)
    with pytest.raises(ValueError):
        HybridSearch(
            embedder=embedder,
            reranker=reranker,
            candidate_pool_size=10,
            rerank_pool_size=20,
        )
