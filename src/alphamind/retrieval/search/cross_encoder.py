"""Cross-encoder reranker backed by sentence-transformers.

A cross-encoder consumes ``(query, passage)`` pairs through one transformer
forward pass and predicts a single relevance score. Unlike dense embeddings
— which encode each text independently and then take a dot product — the
cross-encoder attends to the pair jointly, which is why it produces
materially better final ordering. The cost is throughput: it cannot score
a whole index. The standard recipe, which is what we use, is to retrieve a
candidate pool via dense + BM25 + RRF, then rerank that pool here.

``sentence-transformers`` is an optional dependency. Install with::

    uv sync --extra rerank

Without it, this module still imports — calling :class:`CrossEncoderReranker`
raises :class:`RerankerError` with a specific install hint so the failure
mode is obvious rather than a generic ``ImportError`` from somewhere deep
in the call stack.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from alphamind.retrieval.search.rerank import (
    RerankCandidate,
    RerankedHit,
    RerankerError,
)

if TYPE_CHECKING:  # pragma: no cover - import-time only typing
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Pinned in ADR 0005. Small (~130 MB), fast on CPU, well-trained on the
# MS MARCO passage-relevance task — a close enough analogue to filing-
# passage relevance to be a sensible default.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEFAULT_BATCH_SIZE = 32


class CrossEncoderReranker:
    """Wraps :class:`sentence_transformers.CrossEncoder` behind the reranker protocol.

    The model is loaded lazily on the first :meth:`rerank` call so import
    of this module — and CLI startup that doesn't need the reranker —
    stays cheap. Inference is dispatched to a worker thread so the event
    loop stays free.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: CrossEncoder | None = None

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> list[RerankedHit]:
        if not candidates:
            return []
        # Model inference is CPU/GPU-bound; hop off the event loop.
        return await asyncio.to_thread(self._rerank_sync, query, list(candidates))

    def _rerank_sync(
        self,
        query: str,
        candidates: list[RerankCandidate],
    ) -> list[RerankedHit]:
        model = self._load_model()
        pairs: list[list[str]] = [[query, cand.text] for cand in candidates]
        try:
            raw_scores: Any = model.predict(pairs, batch_size=self._batch_size)
        except Exception as exc:
            raise RerankerError(f"cross-encoder inference failed: {exc}") from exc

        # ``CrossEncoder.predict`` returns a NumPy array of shape (N,).
        scores = [float(s) for s in raw_scores]
        if len(scores) != len(candidates):
            raise RerankerError(
                f"cross-encoder returned {len(scores)} scores for {len(candidates)} pairs"
            )

        # Sort by score descending; stable tie-break by input order.
        scored: list[tuple[int, RerankedHit]] = [
            (idx, RerankedHit(chunk_id=cand.chunk_id, score=score))
            for idx, (cand, score) in enumerate(zip(candidates, scores, strict=True))
        ]
        scored.sort(key=lambda pair: (-pair[1].score, pair[0]))
        return [hit for _, hit in scored]

    def _load_model(self) -> CrossEncoder:
        if self._model is not None:
            return self._model
        try:
            # Deferred so the dep is genuinely optional — callers that never
            # construct a CrossEncoderReranker never trigger this import.
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised only when extra absent
            raise RerankerError(
                "CrossEncoderReranker requires the 'rerank' optional extra: "
                "install with `uv sync --extra rerank`."
            ) from exc

        logger.info("loading cross-encoder model %s", self._model_name)
        self._model = CrossEncoder(self._model_name)
        return self._model


__all__ = ["DEFAULT_BATCH_SIZE", "DEFAULT_MODEL", "CrossEncoderReranker"]
