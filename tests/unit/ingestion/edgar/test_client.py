"""Unit tests for the SEC EDGAR HTTP client."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import pytest
import respx

from alphamind.ingestion.edgar.client import (
    SEC_DATA_BASE,
    SEC_WWW_BASE,
    EdgarClient,
    TokenBucketLimiter,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def user_agent() -> str:
    return "AlphaMind test test@example.com"


async def test_token_bucket_rate_limits_excess_calls() -> None:
    limiter = TokenBucketLimiter(rate=5, interval=1.0)

    started = time.monotonic()
    # Exhaust the bucket (5 tokens), then a 6th call must wait.
    for _ in range(6):
        await limiter.acquire()
    elapsed = time.monotonic() - started

    # The 6th token refills at 5 tokens/sec, so the extra wait is ~0.2s.
    # Allow generous slack to keep the test non-flaky on loaded CI hosts.
    assert elapsed >= 0.15


async def test_token_bucket_rejects_invalid_rates() -> None:
    with pytest.raises(ValueError):
        TokenBucketLimiter(rate=0)
    with pytest.raises(ValueError):
        TokenBucketLimiter(rate=1, interval=0.0)


async def test_get_company_tickers_returns_parsed_json(user_agent: str) -> None:
    payload: dict[str, Any] = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    }

    async with respx.mock(assert_all_called=True) as mock:
        route = mock.get(f"{SEC_WWW_BASE}/files/company_tickers.json").respond(json=payload)

        async with EdgarClient(user_agent=user_agent) as client:
            result = await client.get_company_tickers()

        assert result == payload
        assert route.called
        request = route.calls.last.request
        assert request.headers["User-Agent"] == user_agent


async def test_get_submissions_pads_cik_and_hits_data_host(user_agent: str) -> None:
    payload: dict[str, Any] = {"cik": 320193, "name": "Apple Inc.", "filings": {"recent": {}}}

    async with respx.mock(assert_all_called=True) as mock:
        route = mock.get(f"{SEC_DATA_BASE}/submissions/CIK0000320193.json").respond(json=payload)

        async with EdgarClient(user_agent=user_agent) as client:
            result = await client.get_submissions("320193")

        assert result["cik"] == 320193
        assert route.called


async def test_retries_on_retryable_status_then_succeeds(user_agent: str) -> None:
    url = f"{SEC_DATA_BASE}/submissions/CIK0000320193.json"

    async with respx.mock(assert_all_called=False) as mock:
        route = mock.get(url).mock(
            side_effect=[
                httpx.Response(429, json={"error": "rate limited"}),
                httpx.Response(503, json={"error": "bad gateway"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )

        async with EdgarClient(
            user_agent=user_agent,
            max_retries=5,
            rate=100,  # disable practical rate limiting for this test
        ) as client:
            result = await client.get_submissions("320193")

        assert result == {"ok": True}
        assert route.call_count == 3


async def test_non_retryable_4xx_raises_immediately(user_agent: str) -> None:
    url = f"{SEC_DATA_BASE}/submissions/CIK0000000001.json"

    async with respx.mock() as mock:
        mock.get(url).respond(404, json={"error": "not found"})

        async with EdgarClient(user_agent=user_agent, rate=100) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.get_submissions("1")


async def test_concurrent_requests_stay_within_rate(user_agent: str) -> None:
    """Ten concurrent requests under a 5 req/s limit must take at least ~1s."""

    url = f"{SEC_WWW_BASE}/files/company_tickers.json"

    async with respx.mock() as mock:
        mock.get(url).respond(json={})

        async with EdgarClient(user_agent=user_agent, rate=5, interval=1.0) as client:
            started = time.monotonic()
            await asyncio.gather(*(client.get_company_tickers() for _ in range(10)))
            elapsed = time.monotonic() - started

    # 5 tokens in the bucket to start, 5 more produced over ~1s to cover the
    # remaining calls. Accept 0.85s to keep CI non-flaky.
    assert elapsed >= 0.85
