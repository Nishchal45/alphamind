"""Hybrid retrieval over ``filing_chunks``.

The retrieval layer exposes three search functions:

- :func:`dense_search` ranks chunks by cosine similarity of their stored
  ``embedding`` against the query's embedding. Backed by the HNSW index
  from migration ``0007_retrieval_indexes``.
- :func:`bm25_search` ranks chunks by ``ts_rank_cd`` over the generated
  ``text_tsv`` column. Backed by the GIN index from the same migration.
- :func:`hybrid_search` runs both searches and fuses the rankings via
  Reciprocal Rank Fusion.

All three accept the same set of optional filters: ``as_of_date`` (no
filings dated after it — critical for honest backtests), ``cik``,
``form_types``, and ``section_labels``.

The functions take an :class:`AsyncSession`; pass one from
:func:`alphamind.db.session.session_scope` at call sites that own a unit
of work, or hand-craft a session in a request handler.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date
from typing import Any

from sqlalchemy import ColumnElement, Select, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from alphamind.embeddings.base import Embedder
from alphamind.models.company import Company
from alphamind.models.filing import Filing
from alphamind.models.filing_chunk import FilingChunk
from alphamind.models.filing_document import FilingDocument
from alphamind.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from alphamind.retrieval.results import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 20
DEFAULT_CANDIDATES = 100
TSV_CONFIG = "english"


def _base_select(score_expr: ColumnElement[Any]) -> Select[Any]:
    """Return a SELECT joining chunk → document → filing → company.

    The score expression is appended as the last column so result hydration
    can pull it positionally from each row.
    """

    return (
        select(
            FilingChunk.id,
            FilingChunk.filing_document_id,
            FilingChunk.section_label,
            FilingChunk.section_title,
            FilingChunk.chunk_index,
            FilingChunk.text,
            FilingDocument.filing_id,
            Filing.form,
            Filing.filing_date,
            Filing.accession_number,
            Company.cik,
            Company.ticker,
            Company.name,
            score_expr.label("score"),
        )
        .join(FilingDocument, FilingDocument.id == FilingChunk.filing_document_id)
        .join(Filing, Filing.id == FilingDocument.filing_id)
        .join(Company, Company.id == Filing.company_id)
    )


def _apply_filters(
    stmt: Select[Any],
    *,
    as_of_date: date | None,
    cik: str | None,
    form_types: frozenset[str] | None,
    section_labels: frozenset[str] | None,
) -> Select[Any]:
    """Add the shared retrieval filters to ``stmt``.

    Exposed at module scope so tests can compile the resulting SQL and
    assert against the WHERE clauses without booting a database.
    """

    if as_of_date is not None:
        stmt = stmt.where(Filing.filing_date <= as_of_date)
    if cik is not None:
        stmt = stmt.where(Company.cik == cik.strip().zfill(10))
    if form_types is not None:
        stmt = stmt.where(Filing.form.in_(form_types))
    if section_labels is not None:
        stmt = stmt.where(FilingChunk.section_label.in_(section_labels))
    return stmt


def _row_to_result(row: Any, *, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=row[0],
        filing_document_id=row[1],
        section_label=row[2],
        section_title=row[3],
        chunk_index=row[4],
        text=row[5],
        filing_id=row[6],
        form=row[7],
        filing_date=row[8],
        accession_number=row[9],
        cik=row[10],
        ticker=row[11],
        company_name=row[12],
        score=score,
    )


async def dense_search(
    *,
    session: AsyncSession,
    embedder: Embedder,
    query: str,
    k: int = DEFAULT_TOP_K,
    as_of_date: date | None = None,
    cik: str | None = None,
    form_types: frozenset[str] | None = None,
    section_labels: frozenset[str] | None = None,
) -> list[RetrievalResult]:
    """Top-K chunks by cosine similarity to ``query``'s embedding.

    The returned ``score`` is ``1 - cosine_distance`` so higher is better
    — consistent with the other search functions. Chunks whose
    ``embedding`` column is NULL are skipped (the HNSW index would skip
    them too, but the explicit filter documents intent).
    """

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    [query_vec] = await embedder.embed([query])

    distance = FilingChunk.embedding.cosine_distance(query_vec)
    stmt = _base_select(distance)
    stmt = _apply_filters(
        stmt,
        as_of_date=as_of_date,
        cik=cik,
        form_types=form_types,
        section_labels=section_labels,
    )
    stmt = stmt.where(FilingChunk.embedding.is_not(None)).order_by(distance.asc()).limit(k)

    rows = (await session.execute(stmt)).all()
    return [_row_to_result(row, score=1.0 - float(row[-1])) for row in rows]


async def bm25_search(
    *,
    session: AsyncSession,
    query: str,
    k: int = DEFAULT_TOP_K,
    as_of_date: date | None = None,
    cik: str | None = None,
    form_types: frozenset[str] | None = None,
    section_labels: frozenset[str] | None = None,
) -> list[RetrievalResult]:
    """Top-K chunks by ``ts_rank_cd`` against a Postgres tsquery built from ``query``.

    ``plainto_tsquery`` is used so callers can pass natural-language input
    (it strips noise, applies the same stemming as the generated column,
    and ANDs the remaining lexemes). The returned ``score`` is the raw
    ``ts_rank_cd`` value; magnitudes are not directly comparable to cosine
    similarity, which is exactly why hybrid search uses RRF rather than a
    weighted sum.
    """

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    tsquery = func.plainto_tsquery(TSV_CONFIG, query)
    rank_expr = func.ts_rank_cd(FilingChunk.text_tsv, tsquery)

    stmt = _base_select(rank_expr)
    stmt = _apply_filters(
        stmt,
        as_of_date=as_of_date,
        cik=cik,
        form_types=form_types,
        section_labels=section_labels,
    )
    stmt = stmt.where(FilingChunk.text_tsv.op("@@")(tsquery)).order_by(desc(rank_expr)).limit(k)

    rows = (await session.execute(stmt)).all()
    return [_row_to_result(row, score=float(row[-1])) for row in rows]


async def hybrid_search(
    *,
    session: AsyncSession,
    embedder: Embedder,
    query: str,
    k: int = DEFAULT_TOP_K,
    dense_candidates: int = DEFAULT_CANDIDATES,
    bm25_candidates: int = DEFAULT_CANDIDATES,
    rrf_k: int = DEFAULT_RRF_K,
    as_of_date: date | None = None,
    cik: str | None = None,
    form_types: frozenset[str] | None = None,
    section_labels: frozenset[str] | None = None,
) -> list[RetrievalResult]:
    """Top-K chunks fused from dense and BM25 rankings via RRF.

    Pulls ``dense_candidates`` from :func:`dense_search` and
    ``bm25_candidates`` from :func:`bm25_search`, fuses by RRF, and returns
    the top ``k``. Each result carries the original per-retriever ranks
    (``dense_rank``, ``bm25_rank``) for debuggability; ``None`` means the
    chunk did not appear in that retriever's candidate set.
    """

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    dense = await dense_search(
        session=session,
        embedder=embedder,
        query=query,
        k=dense_candidates,
        as_of_date=as_of_date,
        cik=cik,
        form_types=form_types,
        section_labels=section_labels,
    )
    bm25 = await bm25_search(
        session=session,
        query=query,
        k=bm25_candidates,
        as_of_date=as_of_date,
        cik=cik,
        form_types=form_types,
        section_labels=section_labels,
    )

    dense_ids = [r.chunk_id for r in dense]
    bm25_ids = [r.chunk_id for r in bm25]
    fused = reciprocal_rank_fusion([dense_ids, bm25_ids], k=rrf_k)

    dense_rank = {cid: i + 1 for i, cid in enumerate(dense_ids)}
    bm25_rank = {cid: i + 1 for i, cid in enumerate(bm25_ids)}

    # Result hydration: prefer the dense-side object since dense_search
    # populated it most recently; fall back to bm25 for chunks only the
    # keyword side surfaced.
    by_id: dict[int, RetrievalResult] = {r.chunk_id: r for r in bm25}
    by_id.update({r.chunk_id: r for r in dense})

    results: list[RetrievalResult] = []
    for chunk_id, fused_score in fused[:k]:
        base = by_id[chunk_id]
        results.append(
            replace(
                base,
                score=fused_score,
                dense_rank=dense_rank.get(chunk_id),
                bm25_rank=bm25_rank.get(chunk_id),
            )
        )

    logger.info(
        "hybrid_search query=%r dense=%d bm25=%d fused=%d returned=%d",
        query,
        len(dense),
        len(bm25),
        len(fused),
        len(results),
    )
    return results


__all__ = [
    "DEFAULT_CANDIDATES",
    "DEFAULT_TOP_K",
    "bm25_search",
    "dense_search",
    "hybrid_search",
]
