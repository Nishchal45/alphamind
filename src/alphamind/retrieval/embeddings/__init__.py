"""Dense embeddings for filing chunks.

The :class:`Embedder` protocol is the only thing call sites (the embed
service, the search pipeline) depend on. Concrete implementations:

- :class:`DeterministicHashEmbedder` — seeded by a SHA-256 of the input.
  Used in tests and as the default in development so the project can run
  end-to-end without downloading a model.
- :class:`GeminiEmbedder` — calls Google's ``gemini-embedding-001`` REST
  endpoint. Free tier; ``GOOGLE_API_KEY`` required. Truncates the
  Matryoshka output to :data:`alphamind.models.filing_chunk.EMBEDDING_DIM`
  and re-normalises.

The factory in :func:`get_embedder` reads ``EMBEDDING_BACKEND`` from config
and returns a singleton. Swapping the backend is a config change.
:func:`dispose_embedder` closes any network resources owned by the
backend; the CLIs call it at shutdown.
"""

from __future__ import annotations

from alphamind.retrieval.embeddings.base import Embedder, EmbedderError
from alphamind.retrieval.embeddings.deterministic import DeterministicHashEmbedder
from alphamind.retrieval.embeddings.factory import dispose_embedder, get_embedder
from alphamind.retrieval.embeddings.gemini import GeminiEmbedder
from alphamind.retrieval.embeddings.service import (
    EmbeddingResult,
    embed_chunks_for_filing,
)

__all__ = [
    "DeterministicHashEmbedder",
    "Embedder",
    "EmbedderError",
    "EmbeddingResult",
    "GeminiEmbedder",
    "dispose_embedder",
    "embed_chunks_for_filing",
    "get_embedder",
]
