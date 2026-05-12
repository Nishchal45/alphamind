"""Embedder backed by Google's generativelanguage REST endpoint.

We talk to the REST API directly rather than via ``google-genai`` — the
project already standardises on ``httpx + tenacity + respx`` for outbound
HTTP (see :class:`alphamind.ingestion.edgar.client`) and that stack covers
retries, rate limiting, and mocked tests without another SDK on the dep
tree.

Why ``gemini-embedding-001`` rather than ``text-embedding-004``: the
former is a Matryoshka model, so ``outputDimensionality`` truncates the
returned vector. We pin the dimension to :data:`EMBEDDING_DIM` (384,
chosen in ADR 0005 to match the eventual ``bge-small-en-v1.5`` backend),
which keeps the ``filing_chunks`` schema stable across embedder swaps.
Truncated Matryoshka vectors are not unit-norm; per Google's guidance we
re-normalise so cosine similarity stays well defined.

The free tier ceiling is 1500 RPM. The default rate-limit below stays
comfortably under it; callers can raise it for paid tiers.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from alphamind.ingestion.edgar.client import TokenBucketLimiter
from alphamind.models.filing_chunk import EMBEDDING_DIM
from alphamind.retrieval.embeddings.base import EmbedderError

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Free-tier limit on ``gemini-embedding-001`` is 1500 RPM. Stay under it
# by default; callers can raise this for paid tiers.
DEFAULT_RATE_PER_SECOND = 20
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 5

# ``batchEmbedContents`` accepts up to 100 inputs per call.
MAX_BATCH_SIZE = 100

# Asymmetric retrieval task type for chunk encoding (documents). The query
# side uses ``RETRIEVAL_QUERY`` so the model emits vectors optimised for
# the same similarity space.
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class _RetryableStatusError(httpx.HTTPStatusError):
    """Marker subclass used by tenacity to trigger a retry."""


class GeminiEmbedder:
    """Async embedder calling Google's ``gemini-embedding-001`` REST endpoint.

    Construct from :func:`alphamind.retrieval.embeddings.factory.get_embedder`
    in normal use; the direct constructor is exposed for tests that need
    to inject a custom :class:`httpx.AsyncBaseTransport` for respx mocking.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-embedding-001",
        dim: int = EMBEDDING_DIM,
        rate: int = DEFAULT_RATE_PER_SECOND,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        task_type: str = DOCUMENT_TASK_TYPE,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self._api_key = api_key
        self._model = model
        self._dim = dim
        self._task_type = task_type
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Content-Type": "application/json"},
            transport=transport,
        )
        self._limiter = TokenBucketLimiter(rate=rate)
        self._max_retries = max_retries

    @property
    def dim(self) -> int:
        return self._dim

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        # Slice oversized inputs into batches of MAX_BATCH_SIZE and issue
        # sequentially. Parallel batches would risk the per-minute quota;
        # sequential keeps rate-limit accounting simple.
        results: list[list[float]] = []
        text_list = list(texts)
        for start in range(0, len(text_list), MAX_BATCH_SIZE):
            batch = text_list[start : start + MAX_BATCH_SIZE]
            results.extend(await self._embed_batch(batch))
        return results

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = f"{GEMINI_API_BASE}/models/{self._model}:batchEmbedContents"
        payload = {
            "requests": [
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": self._task_type,
                    "outputDimensionality": self._dim,
                }
                for text in texts
            ]
        }

        retryer = retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type((httpx.TransportError, _RetryableStatusError)),
            reraise=True,
        )

        @retryer
        async def _do() -> httpx.Response:
            await self._limiter.acquire()
            response = await self._client.post(
                url,
                params={"key": self._api_key},
                json=payload,
            )
            if response.status_code in _RETRYABLE_STATUS:
                raise _RetryableStatusError(
                    f"retryable status {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return response

        try:
            response = await _do()
        except httpx.HTTPStatusError as exc:
            raise EmbedderError(
                f"gemini embed failed ({exc.response.status_code}): {exc.response.text[:200]}"
            ) from exc
        except httpx.TransportError as exc:
            raise EmbedderError(f"gemini embed transport error: {exc}") from exc

        raw = _parse_batch_response(response.json(), expected=len(texts))
        # Matryoshka-truncated vectors are not unit-norm. Re-normalise so the
        # cosine-similarity contract in :class:`Embedder` still holds.
        return [_l2_normalise(vec) for vec in raw]


def _parse_batch_response(payload: Any, *, expected: int) -> list[list[float]]:
    """Extract ``[[float, ...], ...]`` from a batchEmbedContents response.

    Defensive: Google's REST layer occasionally returns extra fields or
    omits ``values`` when an input is too long. Any deviation from the
    expected shape becomes an :class:`EmbedderError` so the caller sees a
    specific failure rather than a generic ``KeyError``.
    """

    if not isinstance(payload, dict):
        raise EmbedderError(f"unexpected response payload type: {type(payload).__name__}")
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != expected:
        raise EmbedderError(
            f"expected {expected} embeddings, got "
            f"{len(embeddings) if isinstance(embeddings, list) else 'non-list'}"
        )

    out: list[list[float]] = []
    for i, entry in enumerate(embeddings):
        if not isinstance(entry, dict):
            raise EmbedderError(f"embedding[{i}] is not an object")
        values = entry.get("values")
        if not isinstance(values, list) or not all(isinstance(v, int | float) for v in values):
            raise EmbedderError(f"embedding[{i}] has malformed 'values'")
        out.append([float(v) for v in values])
    return out


def _l2_normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RATE_PER_SECOND",
    "DEFAULT_TIMEOUT_SECONDS",
    "DOCUMENT_TASK_TYPE",
    "GEMINI_API_BASE",
    "MAX_BATCH_SIZE",
    "GeminiEmbedder",
]
