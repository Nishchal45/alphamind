"""Hybrid retrieval over filing chunks.

Public surface:

- :class:`RetrievalResult` — one ranked chunk with enough context to cite.
- :func:`reciprocal_rank_fusion` — pure function that fuses ranked lists.
- :func:`dense_search` — pgvector cosine ANN over ``filing_chunks.embedding``.
- :func:`bm25_search` — ``ts_rank_cd`` BM25-style search over the generated
  ``text_tsv`` column.
- :func:`hybrid_search` — runs both searches and fuses the results via RRF.

All three search functions accept optional filters: ``as_of_date`` (no
filings dated after it — required for honest backtests), ``cik``,
``form_types``, and ``section_labels``.
"""

from __future__ import annotations

from alphamind.retrieval.fusion import (
    DEFAULT_RRF_K,
    reciprocal_rank_fusion,
)
from alphamind.retrieval.results import RetrievalResult
from alphamind.retrieval.service import (
    DEFAULT_CANDIDATES,
    DEFAULT_TOP_K,
    bm25_search,
    dense_search,
    hybrid_search,
)

__all__ = [
    "DEFAULT_CANDIDATES",
    "DEFAULT_RRF_K",
    "DEFAULT_TOP_K",
    "RetrievalResult",
    "bm25_search",
    "dense_search",
    "hybrid_search",
    "reciprocal_rank_fusion",
]
