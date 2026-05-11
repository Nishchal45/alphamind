"""Retrieval result dataclass shared by every search function.

A :class:`RetrievalResult` carries enough context to cite the chunk back
to a source filing without a second round-trip to the database. The
``score`` is whatever the producer is ranking by — cosine similarity for
:func:`dense_search`, ``ts_rank_cd`` for :func:`bm25_search`, the fused
RRF score for :func:`hybrid_search`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked chunk plus the filing context needed to cite it."""

    chunk_id: int
    filing_document_id: int
    filing_id: int

    cik: str
    ticker: str | None
    company_name: str
    form: str
    filing_date: date
    accession_number: str

    section_label: str
    section_title: str | None
    chunk_index: int
    text: str

    # Ranking signal. Interpretation depends on the producer; see module
    # docstring on each search function for specifics.
    score: float

    # Per-retriever ranks (1-indexed) carried through hybrid_search for
    # debuggability. ``None`` when the chunk did not appear in that
    # retriever's top-K. Always ``None`` for the single-retriever
    # functions.
    dense_rank: int | None = None
    bm25_rank: int | None = None


__all__ = ["RetrievalResult"]
