"""Tests for :class:`FundamentalsSpecialist`.

The fundamentals specialist is the canonical :class:`SpecialistBase`
implementation, so these tests cover the full base pipeline:
augmentation, retrieval, hydration, parsing, citation validation, and
fallback. Risk and sentiment add slim tests for their domain-specific
surface; they inherit the same plumbing.
"""

from __future__ import annotations

import json

import pytest

from alphamind.agents.specialists.fundamentals import FundamentalsSpecialist
from tests.unit.agents.conftest import ScriptedLLMClient
from tests.unit.agents.specialists.conftest import (
    FakeSearch,
    make_chunk,
    make_hit,
    make_state,
    patch_session_scope,
)

pytestmark = pytest.mark.asyncio


async def test_fundamentals_produces_cited_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = [make_hit(chunk_id=1), make_hit(chunk_id=2)]
    patch_session_scope(monkeypatch, [make_chunk(chunk_id=1), make_chunk(chunk_id=2)])

    search = FakeSearch(hits)
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
    update = await spec(make_state(query="what are NVDA's segment revenues?"))
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
    patch_session_scope(monkeypatch, [make_chunk(chunk_id=1)])
    search = FakeSearch([make_hit(chunk_id=1)])
    llm = ScriptedLLMClient([json.dumps({"summary": "s", "claims": []})])

    spec = FundamentalsSpecialist(llm=llm, search=search)  # type: ignore[arg-type]
    await spec(make_state(query="China exposure"))

    assert search.last_query is not None
    assert "China exposure" in search.last_query
    # Augmentation terms are appended.
    assert "revenue" in search.last_query


async def test_fundamentals_drops_claims_with_out_of_range_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_session_scope(monkeypatch, [make_chunk(chunk_id=1)])
    search = FakeSearch([make_hit(chunk_id=1)])
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
    update = await spec(make_state())
    claims = update["specialist_reports"][0].claims
    assert len(claims) == 1
    assert claims[0].text == "Good one."


async def test_fundamentals_returns_empty_report_when_retrieval_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_session_scope(monkeypatch, [])
    search = FakeSearch([])
    llm = ScriptedLLMClient([])  # should not be called

    spec = FundamentalsSpecialist(llm=llm, search=search)  # type: ignore[arg-type]
    update = await spec(make_state())
    report = update["specialist_reports"][0]

    assert report.claims == ()
    assert report.citations == ()
    assert "no findings" in report.summary
    assert llm.calls == []


async def test_fundamentals_falls_back_on_malformed_llm_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_session_scope(monkeypatch, [make_chunk(chunk_id=1)])
    search = FakeSearch([make_hit(chunk_id=1)])
    llm = ScriptedLLMClient(["this is not JSON"])

    spec = FundamentalsSpecialist(llm=llm, search=search)  # type: ignore[arg-type]
    update = await spec(make_state())
    report = update["specialist_reports"][0]
    assert report.claims == ()
    assert "no findings" in report.summary
