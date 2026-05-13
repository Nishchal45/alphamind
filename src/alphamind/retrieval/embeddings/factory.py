"""Factory that returns the configured embedder as a singleton.

Concrete backends are owned by :func:`get_embedder`. The factory caches a
single instance per process; backends that own async resources (e.g. an
httpx client) get closed by :func:`dispose_embedder` at shutdown, mirroring
:func:`alphamind.db.session.dispose_engine`.
"""

from __future__ import annotations

from functools import lru_cache

from alphamind.config import get_settings
from alphamind.retrieval.embeddings.base import Embedder, EmbedderError
from alphamind.retrieval.embeddings.deterministic import DeterministicHashEmbedder
from alphamind.retrieval.embeddings.gemini import GeminiEmbedder


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return the process-wide embedder instance."""

    settings = get_settings()
    backend = settings.embedding_backend.lower()

    if backend == "deterministic":
        return DeterministicHashEmbedder()

    if backend == "gemini":
        if not settings.google_api_key:
            raise EmbedderError("embedding_backend='gemini' requires GOOGLE_API_KEY to be set")
        return GeminiEmbedder(
            api_key=settings.google_api_key,
            model=settings.gemini_embedding_model,
        )

    raise EmbedderError(f"unsupported embedding backend: {backend!r}")


async def dispose_embedder() -> None:
    """Close the cached embedder's resources and clear the cache.

    Safe to call when no embedder has been constructed yet. The
    deterministic backend has no async resources; only the network-backed
    backends need this.
    """

    if get_embedder.cache_info().currsize == 0:
        return
    embedder = get_embedder()
    aclose = getattr(embedder, "aclose", None)
    if aclose is not None:
        await aclose()
    get_embedder.cache_clear()


__all__ = ["dispose_embedder", "get_embedder"]
