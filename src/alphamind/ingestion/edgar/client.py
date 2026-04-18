"""Async HTTP client for SEC EDGAR endpoints.

SEC fair-access rules drive three design points:

1. A ``User-Agent`` header identifying the application and an email contact
   is required — unidentified requests are rejected with 403.
2. Total request rate is capped at 10 per second across a client IP.
3. Transient 429 and 5xx responses are common under load; consumers are
   expected to back off rather than hammer retries.

The :class:`EdgarClient` enforces a local token-bucket limiter (default 8 req/s,
safely under the ceiling) and wraps each request in an exponential-backoff
retry via ``tenacity``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import TracebackType
from typing import Any, Self

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from alphamind.config import get_settings

logger = logging.getLogger(__name__)

SEC_WWW_BASE = "https://www.sec.gov"
SEC_DATA_BASE = "https://data.sec.gov"

# SEC's published ceiling is 10 req/s. Default slightly under to leave
# headroom for bursty user code and concurrent adapters.
DEFAULT_RATE_PER_SECOND = 8
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 5

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class TokenBucketLimiter:
    """Async token-bucket rate limiter.

    Parameters
    ----------
    rate:
        Tokens added per ``interval`` seconds (also the bucket capacity).
    interval:
        Window over which ``rate`` tokens are produced. Defaults to 1 second.
    """

    def __init__(self, rate: int, interval: float = 1.0) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if interval <= 0:
            raise ValueError("interval must be positive")
        self._capacity = float(rate)
        self._tokens = float(rate)
        self._refill_per_second = float(rate) / float(interval)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a single token is available, then consume it."""

        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed * self._refill_per_second,
            )
            self._last_refill = now

            if self._tokens < 1:
                wait = (1.0 - self._tokens) / self._refill_per_second
                await asyncio.sleep(wait)
                now = time.monotonic()
                self._last_refill = now
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


class RetryableStatusError(httpx.HTTPStatusError):
    """Marker subclass used by tenacity to trigger a retry."""


class EdgarClient:
    """Async client for the public SEC EDGAR JSON endpoints."""

    def __init__(
        self,
        *,
        user_agent: str | None = None,
        rate: int = DEFAULT_RATE_PER_SECOND,
        interval: float = 1.0,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        agent = user_agent or get_settings().sec_user_agent
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json",
            },
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )
        self._limiter = TokenBucketLimiter(rate=rate, interval=interval)
        self._max_retries = max_retries

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

    async def _get(self, url: str) -> httpx.Response:
        """Issue a rate-limited GET with retries on transient failures."""

        retryer = retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type((httpx.TransportError, RetryableStatusError)),
            reraise=True,
        )

        @retryer
        async def _do() -> httpx.Response:
            await self._limiter.acquire()
            response = await self._client.get(url)
            if response.status_code in _RETRYABLE_STATUS:
                raise RetryableStatusError(
                    f"retryable status {response.status_code} for {url}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return response

        return await _do()

    async def get_company_tickers(self) -> dict[str, Any]:
        """Fetch SEC's public ticker-to-CIK map."""

        url = f"{SEC_WWW_BASE}/files/company_tickers.json"
        logger.debug("edgar: fetching ticker map")
        response = await self._get(url)
        return response.json()  # type: ignore[no-any-return]

    async def get_submissions(self, cik: str) -> dict[str, Any]:
        """Fetch recent-submissions metadata for a CIK."""

        padded = cik.strip().zfill(10)
        url = f"{SEC_DATA_BASE}/submissions/CIK{padded}.json"
        logger.debug("edgar: fetching submissions cik=%s", padded)
        response = await self._get(url)
        return response.json()  # type: ignore[no-any-return]


__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RATE_PER_SECOND",
    "DEFAULT_TIMEOUT_SECONDS",
    "EdgarClient",
    "RetryableStatusError",
    "SEC_DATA_BASE",
    "SEC_WWW_BASE",
    "TokenBucketLimiter",
]
