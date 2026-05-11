"""Cross-encoder reranking for hybrid retrieval results.

Public surface:

- :class:`Reranker` — backend protocol.
- :class:`RerankedPassage` — output dataclass.
- :class:`DeterministicReranker` — Jaccard-overlap, model-free.
- :class:`CrossEncoderReranker` — sentence-transformers; requires the
  ``rerank`` extra.
- :func:`rerank_results` — adapts a reranker onto a list of
  :class:`alphamind.retrieval.RetrievalResult`.
- :func:`get_reranker` — config-driven singleton accessor.
"""

from __future__ import annotations

from alphamind.reranking.base import (
    RerankedPassage,
    Reranker,
    RerankerError,
)
from alphamind.reranking.cross_encoder import (
    DEFAULT_MODEL as DEFAULT_CROSS_ENCODER_MODEL,
)
from alphamind.reranking.cross_encoder import (
    CrossEncoderReranker,
)
from alphamind.reranking.deterministic import DeterministicReranker
from alphamind.reranking.factory import dispose_reranker, get_reranker
from alphamind.reranking.service import rerank_results

__all__ = [
    "DEFAULT_CROSS_ENCODER_MODEL",
    "CrossEncoderReranker",
    "DeterministicReranker",
    "RerankedPassage",
    "Reranker",
    "RerankerError",
    "dispose_reranker",
    "get_reranker",
    "rerank_results",
]
