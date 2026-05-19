"""Tests for /healthz and /readyz."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest

from alphamind.api.routes import health as health_module
from tests.unit.api.conftest import build_app

pytestmark = pytest.mark.asyncio


async def test_healthz_returns_ok(asgi_transport_factory: Any) -> None:
    app, _ = build_app([])
    async with asgi_transport_factory(app) as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "detail": None}


async def test_readyz_ok_when_db_reachable(
    monkeypatch: pytest.MonkeyPatch,
    asgi_transport_factory: Any,
) -> None:
    class _FakeSession:
        async def execute(self, _stmt: Any) -> None:
            return None

    @asynccontextmanager
    async def fake_scope() -> Any:
        yield _FakeSession()

    monkeypatch.setattr(health_module, "session_scope", fake_scope)

    app, _ = build_app([])
    async with asgi_transport_factory(app) as client:
        response = await client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_readyz_503_when_db_unreachable(
    monkeypatch: pytest.MonkeyPatch,
    asgi_transport_factory: Any,
) -> None:
    @asynccontextmanager
    async def fake_scope() -> Any:
        raise RuntimeError("db down")
        yield  # pragma: no cover — unreachable, satisfies the generator contract

    monkeypatch.setattr(health_module, "session_scope", fake_scope)

    app, _ = build_app([])
    async with asgi_transport_factory(app) as client:
        response = await client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "down"
    assert "db down" in body["detail"]
