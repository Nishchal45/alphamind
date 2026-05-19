"""Shared fixtures for the API test suite.

The API tests run against a real FastAPI app, but the underlying
:class:`ResearchGraph` is replaced with a stub that yields canned
``(node_name, state_update)`` pairs. This keeps the tests focused on
the HTTP / SSE surface — graph correctness is covered in
:mod:`tests.unit.agents`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from alphamind.agents.state import GraphInput
from alphamind.api.app import create_app
from alphamind.api.dependencies import get_research_graph_dep


class StubResearchGraph:
    """Drop-in for :class:`ResearchGraph`. Yields canned updates."""

    def __init__(self, updates: Sequence[tuple[str, dict[str, Any]]]) -> None:
        self._updates = list(updates)
        self.last_payload: GraphInput | None = None

    async def stream_updates(
        self,
        payload: GraphInput,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        self.last_payload = payload
        for node_name, state_update in self._updates:
            yield node_name, state_update


def build_app(
    stub_updates: Iterable[tuple[str, dict[str, Any]]],
) -> tuple[FastAPI, StubResearchGraph]:
    """Build an app with a stub graph injected via dependency override.

    Returns ``(app, stub)`` so tests can inspect the captured payload
    after the request runs.
    """

    app = create_app()
    stub = StubResearchGraph(list(stub_updates))
    app.dependency_overrides[get_research_graph_dep] = lambda: stub
    return app, stub


@pytest.fixture
def asgi_transport_factory() -> Any:
    """Return a helper that wraps ``AsyncClient`` around a FastAPI app.

    Using ``ASGITransport`` skips uvicorn entirely — the requests are
    handled in-process. Lifespan still runs (so app.state is populated)
    because ``AsyncClient(transport=...)`` triggers startup / shutdown
    around its context manager.
    """

    def _factory(app: FastAPI) -> AsyncClient:
        return AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        )

    return _factory
