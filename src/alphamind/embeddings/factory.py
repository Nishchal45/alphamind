"""Factory that returns the configured embedder as a process-wide singleton."""

from __future__ import annotations

from functools import lru_cache

from alphamind.config import get_settings
from alphamind.embeddings.base import Embedder, EmbedderError
from alphamind.embeddings.deterministic import DeterministicEmbedder
from alphamind.embeddings.gemini import GeminiEmbedder


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return the process-wide embedder instance."""

    settings = get_settings()
    backend = settings.embedder_backend.lower()

    if backend == "deterministic":
        return DeterministicEmbedder(dimension=settings.embedder_dimension)

    if backend == "gemini":
        if not settings.google_api_key:
            raise EmbedderError("embedder_backend='gemini' requires GOOGLE_API_KEY to be set")
        return GeminiEmbedder(
            api_key=settings.google_api_key,
            model=settings.gemini_embedding_model,
            dimension=settings.embedder_dimension,
        )

    raise EmbedderError(f"unsupported embedder backend: {backend!r}")


async def dispose_embedder() -> None:
    """Close the cached embedder's HTTP client and clear the cache.

    Mirrors :func:`alphamind.db.session.dispose_engine`. Safe to call when
    no embedder has been constructed yet.
    """

    if get_embedder.cache_info().currsize == 0:
        return
    embedder = get_embedder()
    aclose = getattr(embedder, "aclose", None)
    if aclose is not None:
        await aclose()
    get_embedder.cache_clear()


__all__ = ["dispose_embedder", "get_embedder"]
