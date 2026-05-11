"""Text-to-vector encoding for filing chunks.

Public surface:

- :class:`Embedder` — backend protocol.
- :class:`DeterministicEmbedder` — hash-based, model-free, deterministic.
- :func:`get_embedder` — config-driven singleton accessor.
"""

from __future__ import annotations

from alphamind.embeddings.base import Embedder, EmbedderError, Vector
from alphamind.embeddings.deterministic import (
    DEFAULT_DIMENSION,
    DeterministicEmbedder,
)
from alphamind.embeddings.factory import dispose_embedder, get_embedder
from alphamind.embeddings.gemini import GeminiEmbedder
from alphamind.embeddings.service import (
    DEFAULT_BATCH_SIZE,
    EmbedBatchResult,
    EmbedIngestResult,
    embed_chunks_for_cik,
    embed_chunks_for_document,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DIMENSION",
    "DeterministicEmbedder",
    "EmbedBatchResult",
    "EmbedIngestResult",
    "Embedder",
    "EmbedderError",
    "GeminiEmbedder",
    "Vector",
    "dispose_embedder",
    "embed_chunks_for_cik",
    "embed_chunks_for_document",
    "get_embedder",
]
