"""Embedder backed by Google's ``generativelanguage`` REST endpoint.

This implementation talks directly to the REST API rather than pulling in
``google-genai`` — the project already standardises on httpx + tenacity +
respx for outbound HTTP (see :class:`alphamind.ingestion.edgar.client`)
and that stack covers everything we need (retries, rate limiting, mocked
tests) without another SDK on the dep tree.

Auth uses the API key as a query parameter, which is what the free-tier
``generativelanguage.googleapis.com`` endpoints accept.

Batching is done via ``:batchEmbedContents`` — up to 100 inputs per call
on the free tier — and the embedder slices oversized inputs into chunks
of :data:`MAX_BATCH_SIZE` transparently so callers do not need to know
the limit.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any, Self

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from alphamind.embeddings.base import EmbedderError, Vector
from alphamind.ingestion.edgar.client import TokenBucketLimiter

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Free-tier ``text-embedding-004`` limit: 1500 RPM. The default below
# stays comfortably under it; callers can raise it for paid tiers.
DEFAULT_RATE_PER_SECOND = 20
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 5

# ``batchEmbedContents`` accepts up to 100 inputs per call.
MAX_BATCH_SIZE = 100

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class _RetryableStatusError(httpx.HTTPStatusError):
    """Marker subclass used by tenacity to trigger a retry."""


class GeminiEmbedder:
    """Async embedder calling Google's ``text-embedding-004`` REST endpoint.

    Construct from :func:`alphamind.embeddings.factory.get_embedder` in
    normal use; the direct constructor is exposed for tests that need to
    inject a custom :class:`httpx.AsyncBaseTransport` for respx mocking.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-004",
        dimension: int = 768,
        rate: int = DEFAULT_RATE_PER_SECOND,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        self._model_name = f"gemini:{model}"
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Content-Type": "application/json"},
            transport=transport,
        )
        self._limiter = TokenBucketLimiter(rate=rate)
        self._max_retries = max_retries

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

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

    async def embed(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []

        # Slice oversized inputs into batches of MAX_BATCH_SIZE and issue
        # them sequentially. Concurrent batches would risk the per-minute
        # quota; sequential keeps rate-limit accounting simple.
        results: list[Vector] = []
        for start in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[start : start + MAX_BATCH_SIZE]
            results.extend(await self._embed_batch(batch))
        return results

    async def _embed_batch(self, texts: list[str]) -> list[Vector]:
        url = f"{GEMINI_API_BASE}/models/{self._model}:batchEmbedContents"
        payload = {
            "requests": [
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": text}]},
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
            # 4xx other than 429: not retryable, surface as embedder error.
            raise EmbedderError(
                f"gemini embed failed ({exc.response.status_code}): {exc.response.text[:200]}"
            ) from exc
        except httpx.TransportError as exc:
            raise EmbedderError(f"gemini embed transport error: {exc}") from exc

        return _parse_batch_response(response.json(), expected=len(texts))


def _parse_batch_response(payload: Any, *, expected: int) -> list[Vector]:
    """Extract ``[[float, ...], ...]`` from a batchEmbedContents response.

    Defensive: Google's REST layer occasionally returns extra fields or
    omits ``values`` when the input is too long. Any deviation from the
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

    out: list[Vector] = []
    for i, entry in enumerate(embeddings):
        if not isinstance(entry, dict):
            raise EmbedderError(f"embedding[{i}] is not an object")
        values = entry.get("values")
        if not isinstance(values, list) or not all(isinstance(v, int | float) for v in values):
            raise EmbedderError(f"embedding[{i}] has malformed 'values'")
        out.append([float(v) for v in values])
    return out


__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RATE_PER_SECOND",
    "DEFAULT_TIMEOUT_SECONDS",
    "GEMINI_API_BASE",
    "MAX_BATCH_SIZE",
    "GeminiEmbedder",
]
