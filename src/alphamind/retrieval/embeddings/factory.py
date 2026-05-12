"""Factory that returns the configured embedder as a singleton."""

from __future__ import annotations

from functools import lru_cache

from alphamind.config import get_settings
from alphamind.retrieval.embeddings.base import Embedder, EmbedderError
from alphamind.retrieval.embeddings.deterministic import DeterministicHashEmbedder


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return the process-wide embedder instance."""

    settings = get_settings()
    backend = settings.embedding_backend.lower()

    if backend == "deterministic":
        return DeterministicHashEmbedder()

    raise EmbedderError(f"unsupported embedding backend: {backend!r}")


__all__ = ["get_embedder"]
