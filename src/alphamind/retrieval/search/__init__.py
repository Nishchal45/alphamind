"""Hybrid retrieval pipeline.

End-user entrypoint is :class:`HybridSearch`. Given a natural-language
query and an as-of date, it returns the top-k chunks from
``filing_chunks`` that:

1. Were filed on or before the as-of date (the time-horizon filter, the
   single most important correctness invariant in the project).
2. Score well under either lexical (BM25 over ``text_tsv``) or dense
   (cosine over ``embedding``) retrieval.
3. Survive a cross-encoder rerank on the merged candidate set.

The components are deliberately split into modules so they can be tested
in isolation:

- :mod:`alphamind.retrieval.search.lexical` — BM25 candidates via
  Postgres ``ts_rank_cd``.
- :mod:`alphamind.retrieval.search.dense` — pgvector ANN candidates.
- :mod:`alphamind.retrieval.search.fusion` — Reciprocal Rank Fusion.
- :mod:`alphamind.retrieval.search.rerank` — :class:`Reranker` protocol +
  Jaccard-overlap stub.
- :mod:`alphamind.retrieval.search.cross_encoder` — sentence-transformers
  cross-encoder backend (optional ``rerank`` extra).
- :mod:`alphamind.retrieval.search.reranker_factory` — config-driven
  reranker singleton (mirrors the embedder factory).
- :mod:`alphamind.retrieval.search.pipeline` — orchestration.
"""

from __future__ import annotations

from alphamind.retrieval.search.cross_encoder import CrossEncoderReranker
from alphamind.retrieval.search.dense import DenseHit, dense_search
from alphamind.retrieval.search.fusion import FusedHit, reciprocal_rank_fusion
from alphamind.retrieval.search.lexical import LexicalHit, lexical_search
from alphamind.retrieval.search.pipeline import HybridSearch, SearchHit
from alphamind.retrieval.search.rerank import (
    DeterministicReranker,
    RerankCandidate,
    RerankedHit,
    Reranker,
    RerankerError,
)
from alphamind.retrieval.search.reranker_factory import dispose_reranker, get_reranker

__all__ = [
    "CrossEncoderReranker",
    "DenseHit",
    "DeterministicReranker",
    "FusedHit",
    "HybridSearch",
    "LexicalHit",
    "RerankCandidate",
    "RerankedHit",
    "Reranker",
    "RerankerError",
    "SearchHit",
    "dense_search",
    "dispose_reranker",
    "get_reranker",
    "lexical_search",
    "reciprocal_rank_fusion",
]
