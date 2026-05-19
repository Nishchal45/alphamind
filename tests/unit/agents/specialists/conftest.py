"""Shared fakes for specialist tests.

Specialists subclass :class:`SpecialistBase`, which opens a DB session,
runs :class:`HybridSearch`, hydrates the chunk + filing + company graph,
and parses LLM JSON. Real instances of all of that exist in integration
tests; here we exercise the specialist's own logic by injecting fakes
that satisfy the duck-typed interface.

The fakes are in this conftest so each specialist's test file can stay
focused on the specialist-specific surface (name, augmentation, prompt
content, smoke test) without redefining ~80 lines of plumbing.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest

from alphamind.agents.specialists import base as specialist_base_module
from alphamind.agents.state import AgentState
from alphamind.retrieval.search.pipeline import SearchHit


@dataclass(frozen=True, slots=True)
class FakeCompany:
    ticker: str | None
    cik: str
    name: str = "n/a"


@dataclass(frozen=True, slots=True)
class FakeFiling:
    form: str
    company: FakeCompany


@dataclass(frozen=True, slots=True)
class FakeChunk:
    """Duck-typed :class:`FilingChunk` with the attributes the hydrator reads."""

    id: int
    filing_id: int
    filing_date: date
    section: str | None
    text: str
    filing: FakeFiling


class _FakeScalars:
    def __init__(self, items: Sequence[FakeChunk]) -> None:
        self._items = list(items)

    def __iter__(self) -> Any:
        return iter(self._items)


class _FakeResult:
    def __init__(self, items: Sequence[FakeChunk]) -> None:
        self._items = list(items)

    def scalars(self) -> _FakeScalars:
        return _FakeScalars(self._items)


class FakeSession:
    """Async-session double; returns canned chunks from ``execute()``."""

    def __init__(self, chunks: Sequence[FakeChunk]) -> None:
        self._chunks = list(chunks)

    async def execute(self, _stmt: Any) -> _FakeResult:
        return _FakeResult(self._chunks)


class FakeSearch:
    """Drop-in for :class:`HybridSearch`. Records the query it was asked for."""

    def __init__(self, hits: Sequence[SearchHit]) -> None:
        self._hits = list(hits)
        self.last_query: str | None = None

    async def search(
        self,
        _session: Any,
        *,
        query: str,
        as_of: date,
        top_k: int = 10,
    ) -> list[SearchHit]:
        self.last_query = query
        return list(self._hits[:top_k])


def make_hit(
    *,
    chunk_id: int,
    filing_id: int | None = None,
    section: str | None = "Item 7. MD&A",
    text: str | None = None,
    filing_date: date = date(2024, 6, 1),
) -> SearchHit:
    """Build a :class:`SearchHit` with sensible defaults for tests."""
    return SearchHit(
        chunk_id=chunk_id,
        filing_id=filing_id if filing_id is not None else 10 + chunk_id,
        filing_date=filing_date,
        section=section,
        text=text if text is not None else f"chunk {chunk_id} body",
        score=0.9,
    )


def make_chunk(
    *,
    chunk_id: int,
    ticker: str = "NVDA",
    form: str = "10-Q",
    section: str | None = "Item 7. MD&A",
    text: str | None = None,
    filing_date: date = date(2024, 6, 1),
) -> FakeChunk:
    """Build a :class:`FakeChunk` with sensible defaults for tests."""
    return FakeChunk(
        id=chunk_id,
        filing_id=10 + chunk_id,
        filing_date=filing_date,
        section=section,
        text=text if text is not None else f"chunk {chunk_id} body",
        filing=FakeFiling(form=form, company=FakeCompany(ticker=ticker, cik="0000000001")),
    )


def patch_session_scope(
    monkeypatch: pytest.MonkeyPatch,
    chunks: Sequence[FakeChunk],
) -> None:
    """Swap ``session_scope`` in the specialist base module with a fake."""

    @asynccontextmanager
    async def fake_scope() -> Any:
        yield FakeSession(chunks)

    monkeypatch.setattr(specialist_base_module, "session_scope", fake_scope)


def make_state(
    *,
    query: str = "test query",
    as_of: date = date(2024, 12, 31),
    top_k: int = 5,
) -> AgentState:
    """Build a minimal :class:`AgentState` for specialist tests."""
    return AgentState(query=query, as_of=as_of, top_k=top_k, specialist_reports=[])
