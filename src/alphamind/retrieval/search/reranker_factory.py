"""Factory returning the configured :class:`Reranker` as a process-wide singleton.

Mirrors :func:`alphamind.retrieval.embeddings.factory.get_embedder`. The
factory caches a single instance per process. Backends that own resources
(loaded models, HTTP clients) get released by :func:`dispose_reranker`.
"""

from __future__ import annotations

from functools import lru_cache

from alphamind.config import get_settings
from alphamind.retrieval.search.cross_encoder import CrossEncoderReranker
from alphamind.retrieval.search.rerank import (
    DeterministicReranker,
    Reranker,
    RerankerError,
)


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

    Concrete backends have no async resources to release today — the
    cross-encoder model is loaded in-process and torn down with the
    interpreter — but this hook exists so callers can shut down
    uniformly across embedding/reranking pairs.
    """

    if get_reranker.cache_info().currsize == 0:
        return
    get_reranker.cache_clear()


__all__ = ["dispose_reranker", "get_reranker"]
