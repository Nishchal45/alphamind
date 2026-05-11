"""Cross-encoder reranker backed by sentence-transformers.

A cross-encoder consumes a ``(query, passage)`` pair through one
transformer forward pass and predicts a single relevance score. Unlike
dense embeddings (which encode each text independently) the cross-encoder
attends to the pair jointly, which is why it produces materially better
final ordering — at the cost of being too slow to score the whole index.
The standard recipe is what this module implements: dense + BM25 retrieve
a candidate pool, the cross-encoder reranks it.

``sentence-transformers`` is an optional dependency: install with
``uv sync --extra rerank`` (or ``pip install 'alphamind[rerank]'``).
Without it, this module still imports — calling :class:`CrossEncoderReranker`
raises a clear :class:`RerankerError` so the failure mode is obvious.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from alphamind.reranking.base import RerankedPassage, RerankerError

if TYPE_CHECKING:  # pragma: no cover - import-time only typing
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Wraps :class:`sentence_transformers.CrossEncoder` behind the protocol.

    The model is loaded lazily on the first :meth:`rerank` call so import
    of this module — and CLI startup that doesn't need the reranker —
    stays cheap. ``model_name`` reflects the requested model identifier
    so persisted results can be traced back to a specific checkpoint.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._requested_model = model_name
        self._batch_size = batch_size
        self._model: CrossEncoder | None = None

    @property
    def model_name(self) -> str:
        return f"cross-encoder:{self._requested_model}"

    def _load_model(self) -> CrossEncoder:
        if self._model is not None:
            return self._model
        try:
            # Deferred import so the cross-encoder dep is truly optional;
            # users without the 'rerank' extra installed never trigger this.
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised only when extra absent
            raise RerankerError(
                "CrossEncoderReranker requires the 'rerank' extra: "
                "install with `uv sync --extra rerank`."
            ) from exc

        logger.info("loading cross-encoder model %s", self._requested_model)
        self._model = CrossEncoder(self._requested_model)
        return self._model

    async def rerank(
        self,
        query: str,
        passages: list[tuple[int, str]],
    ) -> list[RerankedPassage]:
        if not passages:
            return []
        # Model inference is CPU/GPU-bound; hop to a worker thread so the
        # event loop is free for whatever else the caller is doing.
        return await asyncio.to_thread(self._rerank_sync, query, passages)

    def _rerank_sync(
        self,
        query: str,
        passages: list[tuple[int, str]],
    ) -> list[RerankedPassage]:
        model = self._load_model()
        pairs: list[list[str]] = [[query, text] for _, text in passages]
        try:
            raw_scores: Any = model.predict(pairs, batch_size=self._batch_size)
        except Exception as exc:
            raise RerankerError(f"cross-encoder inference failed: {exc}") from exc

        # ``CrossEncoder.predict`` returns a numpy array of shape (N,).
        scores = [float(s) for s in raw_scores]
        if len(scores) != len(passages):
            raise RerankerError(
                f"cross-encoder returned {len(scores)} scores for {len(passages)} pairs"
            )

        scored: list[tuple[int, RerankedPassage]] = [
            (idx, RerankedPassage(chunk_id=chunk_id, score=score))
            for idx, ((chunk_id, _), score) in enumerate(zip(passages, scores, strict=True))
        ]
        scored.sort(key=lambda pair: (-pair[1].score, pair[0]))
        return [pair[1] for pair in scored]


__all__ = ["DEFAULT_MODEL", "CrossEncoderReranker"]
