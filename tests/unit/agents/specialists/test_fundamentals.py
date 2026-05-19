"""Tests for :class:`FundamentalsSpecialist`.

Tests stub out the DB and HybridSearch so they exercise the specialist's
own logic — JSON parsing, claim normalisation, augmentation of the
retrieval query, fallback construction.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest

from alphamind.agents.specialists import base as specialist_base_module
from alphamind.agents.specialists.fundamentals import FundamentalsSpecialist
from alphamind.agents.state import AgentState
from alphamind.retrieval.search.pipeline import SearchHit
from tests.unit.agents.conftest import ScriptedLLMClient

pytestmark = pytest.mark.asyncio


# -- Test doubles -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Company:
    ticker: str | None
    cik: str
    name: str = "n/a"


@dataclass(frozen=True, slots=True)
class _Filing:
    form: str
    company: _Company


@dataclass(frozen=True, slots=True)
class _Chunk:
    id: int
    filing_id: int
    filing_date: date
    section: str | None
    text: str
    filing: _Filing


class _FakeScalars:
    def __init__(self, items: Sequence[_Chunk]) -> None:
        self._items = list(items)

    def __iter__(self) -> Any:
        return iter(self._items)


class _FakeResult:
    def __init__(self, items: Sequence[_Chunk]) -> None:
        self._items = list(items)

    def scalars(self) -> _FakeScalars:
        return _FakeScalars(self._items)


class _FakeSession:
    def __init__(self, chunks: Sequence[_Chunk]) -> None:
        self._chunks = list(chunks)

    async def execute(self, _stmt: Any) -> _FakeResult:
        return _FakeResult(self._chunks)


class _FakeSearch:
    """Returns canned :class:`SearchHit` lists and records the query it saw."""

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


def _hit(*, chunk_id: int, ordinal_filing: int = 1) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        filing_id=10 + ordinal_filing,
        filing_date=date(2024, 6, 1),
        section="Item 7. MD&A",
        text=f"chunk {chunk_id} body about revenue",
        score=0.9,
    )


def _chunk(*, chunk_id: int, ticker: str = "NVDA") -> _Chunk:
    return _Chunk(
        id=chunk_id,
        filing_id=10 + chunk_id,
        filing_date=date(2024, 6, 1),
        section="Item 7. MD&A",
        text=f"chunk {chunk_id} body about revenue",
        filing=_Filing(form="10-Q", company=_Company(ticker=ticker, cik="0000000001")),
    )


def _patch_session_scope(monkeypatch: pytest.MonkeyPatch, chunks: Sequence[_Chunk]) -> None:
    @asynccontextmanager
    async def fake_scope() -> Any:
        yield _FakeSession(chunks)

    monkeypatch.setattr(specialist_base_module, "session_scope", fake_scope)


def _state(query: str = "what are NVDA's segment revenues?") -> AgentState:
    return AgentState(query=query, as_of=date(2024, 12, 31), top_k=5, specialist_reports=[])


# -- Tests --------------------------------------------------------------------


async def test_fundamentals_produces_cited_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = [_hit(chunk_id=1), _hit(chunk_id=2)]
    chunks = [_chunk(chunk_id=1), _chunk(chunk_id=2)]
    _patch_session_scope(monkeypatch, chunks)

    search = _FakeSearch(hits)
    llm = ScriptedLLMClient(
        [
            json.dumps(
                {
                    "summary": "Data-center is the biggest segment.",
                    "claims": [
                        {"text": "Data-center grew YoY.", "citations": [1]},
                        {"text": "Gaming revenue declined.", "citations": [2]},
                    ],
                }
            )
        ]
    )

    spec = FundamentalsSpecialist(llm=llm, search=search)  # type: ignore[arg-type]
    update = await spec(_state())
    reports = update["specialist_reports"]
    assert len(reports) == 1
    report = reports[0]

    assert report.specialist == "fundamentals"
    assert len(report.claims) == 2
    assert {c.text for c in report.claims} == {
        "Data-center grew YoY.",
        "Gaming revenue declined.",
    }
    # Citation ordinals must be in range [1, n_sources].
    assert all(
        1 <= ordinal <= len(report.citations)
        for claim in report.claims
        for ordinal in claim.citations
    )


async def test_fundamentals_augments_retrieval_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_scope(monkeypatch, [_chunk(chunk_id=1)])
    search = _FakeSearch([_hit(chunk_id=1)])
    llm = ScriptedLLMClient([json.dumps({"summary": "s", "claims": []})])

    spec = FundamentalsSpecialist(llm=llm, search=search)  # type: ignore[arg-type]
    await spec(_state(query="China exposure"))

    assert search.last_query is not None
    assert "China exposure" in search.last_query
    # Augmentation terms are appended.
    assert "revenue" in search.last_query


async def test_fundamentals_drops_claims_with_out_of_range_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_scope(monkeypatch, [_chunk(chunk_id=1)])
    search = _FakeSearch([_hit(chunk_id=1)])
    llm = ScriptedLLMClient(
        [
            json.dumps(
                {
                    "summary": "s",
                    "claims": [
                        {"text": "Bad cite.", "citations": [9]},  # out of range
                        {"text": "No cite at all."},
                        {"text": "Good one.", "citations": [1]},
                    ],
                }
            )
        ]
    )

    spec = FundamentalsSpecialist(llm=llm, search=search)  # type: ignore[arg-type]
    update = await spec(_state())
    claims = update["specialist_reports"][0].claims
    assert len(claims) == 1
    assert claims[0].text == "Good one."


async def test_fundamentals_returns_empty_report_when_retrieval_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_scope(monkeypatch, [])
    search = _FakeSearch([])
    llm = ScriptedLLMClient([])  # should not be called

    spec = FundamentalsSpecialist(llm=llm, search=search)  # type: ignore[arg-type]
    update = await spec(_state())
    report = update["specialist_reports"][0]

    assert report.claims == ()
    assert report.citations == ()
    assert "no findings" in report.summary
    assert llm.calls == []


async def test_fundamentals_falls_back_on_malformed_llm_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_scope(monkeypatch, [_chunk(chunk_id=1)])
    search = _FakeSearch([_hit(chunk_id=1)])
    llm = ScriptedLLMClient(["this is not JSON"])

    spec = FundamentalsSpecialist(llm=llm, search=search)  # type: ignore[arg-type]
    update = await spec(_state())
    report = update["specialist_reports"][0]
    assert report.claims == ()
    assert "no findings" in report.summary
