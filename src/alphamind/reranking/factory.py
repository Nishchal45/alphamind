"""Factory returning the configured reranker as a process-wide singleton."""

from __future__ import annotations

from functools import lru_cache

from alphamind.config import get_settings
from alphamind.reranking.base import Reranker, RerankerError
from alphamind.reranking.cross_encoder import CrossEncoderReranker
from alphamind.reranking.deterministic import DeterministicReranker


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Return the process-wide reranker instance."""

    settings = get_settings()
    backend = settings.reranker_backend.lower()

    if backend == "deterministic":
        return DeterministicReranker()

    if backend == "cross_encoder":
        return CrossEncoderReranker(model_name=settings.cross_encoder_model)

    raise RerankerError(f"unsupported reranker backend: {backend!r}")


async def dispose_reranker() -> None:
    """Drop the cached reranker.

    Mirrors :func:`alphamind.embeddings.factory.dispose_embedder`. Safe to
    call when no reranker has been constructed yet. Concrete backends
    have no async resources to release today; this hook exists so callers
    can shut down uniformly across the embedding/reranking pair.
    """

    if get_reranker.cache_info().currsize == 0:
        return
    get_reranker.cache_clear()


__all__ = ["dispose_reranker", "get_reranker"]
