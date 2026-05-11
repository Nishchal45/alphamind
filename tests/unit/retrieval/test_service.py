"""Unit tests for the retrieval service.

The retrieval SQL itself (cosine_distance, ts_rank_cd, GIN/HNSW indexes)
needs a live Postgres with pgvector and so lives in integration tests.
Here we cover the testable seams:

- The shared filter helper builds the right WHERE clauses.
- :func:`hybrid_search` correctly orchestrates dense + bm25 and fuses
  their results — verified by monkey-patching the two sub-searches.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from sqlalchemy import func

from alphamind.retrieval import service as retrieval_service
from alphamind.retrieval.results import RetrievalResult
from alphamind.retrieval.service import (
    _apply_filters,
    _base_select,
    hybrid_search,
)

# pytest-asyncio runs in auto mode (see pyproject.toml), so async tests in
# this file are picked up without an explicit marker. Sync filter-builder
# tests stay sync.


def _compile(stmt: Any) -> str:
    return str(stmt.compile(compile_kwargs={"literal_binds": True}))


def test_apply_filters_no_args_emits_no_where() -> None:
    base = _base_select(func.count())
    stmt = _apply_filters(
        base,
        as_of_date=None,
        cik=None,
        form_types=None,
        section_labels=None,
    )

    sql = _compile(stmt).lower()
    # The base joins are still there; only the filter-derived WHERE
    # clauses should be missing.
    assert "filing_date" not in sql.split("where")[-1] if "where" in sql else True


def test_apply_filters_as_of_date_clauses_filing_date() -> None:
    base = _base_select(func.count())
    stmt = _apply_filters(
        base,
        as_of_date=date(2025, 1, 1),
        cik=None,
        form_types=None,
        section_labels=None,
    )

    sql = _compile(stmt).lower()
    assert "filings.filing_date <=" in sql
    assert "2025-01-01" in sql


def test_apply_filters_cik_is_zero_padded() -> None:
    base = _base_select(func.count())
    stmt = _apply_filters(
        base,
        as_of_date=None,
        cik="320193",  # Apple, 6 chars
        form_types=None,
        section_labels=None,
    )

    sql = _compile(stmt).lower()
    assert "companies.cik = '0000320193'" in sql


def test_apply_filters_form_types_uses_in_clause() -> None:
    base = _base_select(func.count())
    stmt = _apply_filters(
        base,
        as_of_date=None,
        cik=None,
        form_types=frozenset({"10-K", "10-Q"}),
        section_labels=None,
    )

    sql = _compile(stmt).lower()
    assert "filings.form in" in sql
    assert "'10-k'" in sql and "'10-q'" in sql


def test_apply_filters_section_labels_uses_in_clause() -> None:
    base = _base_select(func.count())
    stmt = _apply_filters(
        base,
        as_of_date=None,
        cik=None,
        form_types=None,
        section_labels=frozenset({"item_1a"}),
    )

    sql = _compile(stmt).lower()
    assert "filing_chunks.section_label in" in sql
    assert "'item_1a'" in sql


# --- hybrid_search orchestration -------------------------------------------------


def _make_result(chunk_id: int, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        filing_document_id=10 + chunk_id,
        filing_id=100 + chunk_id,
        cik="0000000001",
        ticker="TST",
        company_name="Test Co",
        form="10-K",
        filing_date=date(2025, 1, 1),
        accession_number=f"0000000001-25-{chunk_id:06d}",
        section_label="preamble",
        section_title=None,
        chunk_index=chunk_id,
        text=f"chunk text {chunk_id}",
        score=score,
    )


class _StubEmbedder:
    model_name = "stub"
    dimension = 2

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0] for _ in texts]


async def test_hybrid_search_fuses_dense_and_bm25_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Dense returns chunks 1, 2, 3. BM25 returns chunks 3, 2, 4.
    # Chunk 2 and 3 appear in both; chunk 3 ranks better overall.
    dense = [
        _make_result(1, 0.9),
        _make_result(2, 0.8),
        _make_result(3, 0.7),
    ]
    bm25 = [
        _make_result(3, 5.0),
        _make_result(2, 4.0),
        _make_result(4, 3.0),
    ]

    async def fake_dense(**_kwargs: Any) -> list[RetrievalResult]:
        return dense

    async def fake_bm25(**_kwargs: Any) -> list[RetrievalResult]:
        return bm25

    monkeypatch.setattr(retrieval_service, "dense_search", fake_dense)
    monkeypatch.setattr(retrieval_service, "bm25_search", fake_bm25)

    results = await hybrid_search(
        session=None,  # type: ignore[arg-type]
        embedder=_StubEmbedder(),
        query="anything",
        k=10,
    )

    # Top results should be chunks that appear in both rankings.
    assert results[0].chunk_id in {2, 3}
    assert results[1].chunk_id in {2, 3}
    # Chunks 1 and 4 (in only one ranking) come after.
    tail_ids = {r.chunk_id for r in results[2:]}
    assert tail_ids == {1, 4}


async def test_hybrid_search_attaches_per_retriever_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense = [_make_result(1, 0.9), _make_result(2, 0.5)]
    bm25 = [_make_result(2, 8.0), _make_result(3, 2.0)]

    async def fake_dense(**_kwargs: Any) -> list[RetrievalResult]:
        return dense

    async def fake_bm25(**_kwargs: Any) -> list[RetrievalResult]:
        return bm25

    monkeypatch.setattr(retrieval_service, "dense_search", fake_dense)
    monkeypatch.setattr(retrieval_service, "bm25_search", fake_bm25)

    results = await hybrid_search(
        session=None,  # type: ignore[arg-type]
        embedder=_StubEmbedder(),
        query="anything",
        k=10,
    )

    by_id = {r.chunk_id: r for r in results}
    # Chunk 1 only in dense at rank 1.
    assert by_id[1].dense_rank == 1
    assert by_id[1].bm25_rank is None
    # Chunk 2 in both: dense rank 2, bm25 rank 1.
    assert by_id[2].dense_rank == 2
    assert by_id[2].bm25_rank == 1
    # Chunk 3 only in bm25 at rank 2.
    assert by_id[3].dense_rank is None
    assert by_id[3].bm25_rank == 2


async def test_hybrid_search_respects_k(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_dense(**_kwargs: Any) -> list[RetrievalResult]:
        return [_make_result(i, 1.0 / i) for i in range(1, 11)]

    async def fake_bm25(**_kwargs: Any) -> list[RetrievalResult]:
        return [_make_result(i + 100, 1.0 / i) for i in range(1, 11)]

    monkeypatch.setattr(retrieval_service, "dense_search", fake_dense)
    monkeypatch.setattr(retrieval_service, "bm25_search", fake_bm25)

    results = await hybrid_search(
        session=None,  # type: ignore[arg-type]
        embedder=_StubEmbedder(),
        query="anything",
        k=3,
    )

    assert len(results) == 3


async def test_hybrid_search_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        await hybrid_search(
            session=None,  # type: ignore[arg-type]
            embedder=_StubEmbedder(),
            query="anything",
            k=0,
        )
