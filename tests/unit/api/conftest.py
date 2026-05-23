"""Shared fixtures for the API test suite.

API tests run against a real FastAPI app, but the underlying compiled
research graph is replaced with a stub that yields canned LangGraph-
shaped updates. Keeps the tests focused on the HTTP / SSE surface —
graph correctness is covered in :mod:`tests.unit.agents`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from alphamind.api.app import create_app
from alphamind.api.dependencies import get_research_graph_dep


class StubResearchGraph:
    """Drop-in for the compiled :class:`ResearchGraph`. Yields canned updates.

    The real graph is a ``CompiledStateGraph`` whose ``astream(...,
    stream_mode="updates")`` emits ``{node_name: state_update}`` dicts.
    This stub replays a scripted sequence with the same shape.
    """

    def __init__(self, updates: Sequence[dict[str, dict[str, Any]]]) -> None:
        self._updates = list(updates)
        self.last_state: dict[str, Any] | None = None
        self.last_stream_mode: str | None = None

    async def astream(
        self,
        state: dict[str, Any],
        *,
        stream_mode: str = "values",
    ) -> AsyncIterator[dict[str, dict[str, Any]]]:
        self.last_state = state
        self.last_stream_mode = stream_mode
        for update in self._updates:
            yield update


def build_app(
    stub_updates: Iterable[dict[str, dict[str, Any]]],
) -> tuple[FastAPI, StubResearchGraph]:
    """Build an app with a stub graph injected via dependency override.

    Returns ``(app, stub)`` so tests can inspect the captured state
    after the request runs.
    """

    app = create_app()
    stub = StubResearchGraph(list(stub_updates))
    app.dependency_overrides[get_research_graph_dep] = lambda: stub
    return app, stub


@pytest.fixture
def asgi_transport_factory() -> Any:
    """Return a helper that wraps ``AsyncClient`` around a FastAPI app.

    Using ``ASGITransport`` skips uvicorn — requests are handled in
    process. Modern httpx doesn't auto-run the FastAPI lifespan, so
    ``app.state.research_graph`` is never populated; tests rely on
    the dependency override instead, which bypasses ``app.state``.
    """

    def _factory(app: FastAPI) -> AsyncClient:
        return AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        )

    return _factory
