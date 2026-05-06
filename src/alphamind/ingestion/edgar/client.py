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
        # No global Accept header: this client hits both JSON endpoints under
        # data.sec.gov and HTML/XML document bodies under www.sec.gov/Archives.
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": agent,
                "Accept-Encoding": "gzip, deflate",
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

    async def get_primary_document(
        self,
        *,
        cik: str,
        accession_number: str,
        primary_document: str,
    ) -> tuple[bytes, str, str]:
        """Fetch the primary document body for a single filing.

        Parameters
        ----------
        cik:
            The filer's CIK. Leading zeros are tolerated and stripped — the
            ``Archives`` URL uses the integer form.
        accession_number:
            EDGAR accession number, with or without dashes
            (e.g. ``"0000320193-24-000123"``).
        primary_document:
            Filename of the primary document inside the filing's archive
            directory (e.g. ``"aapl-20240928.htm"``). Comes from the
            submissions endpoint.

        Returns
        -------
        ``(body, content_type, source_url)`` — raw bytes, the response's
        ``Content-Type`` header (defaults to ``application/octet-stream`` if
        absent), and the canonical EDGAR URL the bytes were fetched from.
        """

        cik_clean = cik.strip().lstrip("0")
        if not cik_clean:
            raise ValueError(f"invalid cik: {cik!r}")
        if not accession_number.strip():
            raise ValueError("accession_number must be non-empty")
        if not primary_document.strip():
            raise ValueError("primary_document must be non-empty")

        accession_dashless = accession_number.replace("-", "")
        url = (
            f"{SEC_WWW_BASE}/Archives/edgar/data/{cik_clean}/"
            f"{accession_dashless}/{primary_document}"
        )
        logger.debug(
            "edgar: fetching primary document cik=%s accession=%s",
            cik_clean,
            accession_number,
        )
        response = await self._get(url)
        content_type = response.headers.get("content-type", "application/octet-stream")
        return response.content, content_type, url


__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RATE_PER_SECOND",
    "DEFAULT_TIMEOUT_SECONDS",
    "SEC_DATA_BASE",
    "SEC_WWW_BASE",
    "EdgarClient",
    "RetryableStatusError",
    "TokenBucketLimiter",
]
