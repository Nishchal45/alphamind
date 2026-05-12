"""Dense embeddings for filing chunks.

The :class:`Embedder` protocol is the only thing call sites (the embed
service, the search pipeline) depend on. Concrete implementations:

- :class:`DeterministicHashEmbedder` — seeded by a SHA-256 of the input.
  Used in tests and as the default in development so the project can run
  end-to-end without downloading a 130 MB sentence-transformer.
- ``SentenceTransformerEmbedder`` — wired in Phase 3, lazily imports
  ``sentence_transformers``. Out of scope for this branch.

The factory in :func:`get_embedder` reads ``EMBEDDING_BACKEND`` from config
and returns a singleton. Swapping the backend is a config change.
"""

from __future__ import annotations

from alphamind.retrieval.embeddings.base import Embedder, EmbedderError
from alphamind.retrieval.embeddings.deterministic import DeterministicHashEmbedder
from alphamind.retrieval.embeddings.factory import get_embedder
from alphamind.retrieval.embeddings.service import (
    EmbeddingResult,
    embed_chunks_for_filing,
)

__all__ = [
    "DeterministicHashEmbedder",
    "Embedder",
    "EmbedderError",
    "EmbeddingResult",
    "embed_chunks_for_filing",
    "get_embedder",
]
