"""``GET /healthz`` smoke tests."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_healthz_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "graph_ready": True}
