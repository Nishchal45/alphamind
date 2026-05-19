"""Tests for the typed agent state and its helpers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from alphamind.agents.state import (
    ALL_SPECIALISTS,
    Citation,
    Claim,
    CriticReport,
    GraphInput,
    SpecialistReport,
    UnsupportedClaim,
    merge_specialist_reports,
)


def _citation(ordinal: int, chunk_id: int) -> Citation:
    return Citation(
        ordinal=ordinal,
        chunk_id=chunk_id,
        filing_id=10 * chunk_id,
        ticker="NVDA",
        form="10-K",
        filing_date=date(2024, 1, 1),
        section="Item 7. MD&A",
        text=f"text-{chunk_id}",
    )


def test_graph_input_rejects_blank_query() -> None:
    with pytest.raises(ValueError):
        GraphInput(query="   ", as_of=date(2024, 1, 1)).to_state()


def test_graph_input_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError):
        GraphInput(query="ok", as_of=date(2024, 1, 1), top_k=0).to_state()


def test_graph_input_seeds_state() -> None:
    state = GraphInput(query="why", as_of=date(2024, 6, 1), top_k=3).to_state()
    assert state["query"] == "why"
    assert state["as_of"] == date(2024, 6, 1)
    assert state["top_k"] == 3
    assert state["specialist_reports"] == []


def test_merge_reducer_concatenates() -> None:
    a = SpecialistReport(specialist="fundamentals", summary="A", claims=(), citations=())
    b = SpecialistReport(specialist="risk", summary="B", claims=(), citations=())
    merged = merge_specialist_reports([a], [b])
    assert [r.specialist for r in merged] == ["fundamentals", "risk"]


def test_merge_reducer_handles_none() -> None:
    a = SpecialistReport(specialist="fundamentals", summary="A", claims=(), citations=())
    assert merge_specialist_reports(None, [a]) == [a]
    assert merge_specialist_reports([a], None) == [a]
    assert merge_specialist_reports(None, None) == []


def test_citation_header_format() -> None:
    c = _citation(ordinal=2, chunk_id=42)
    assert c.header() == "[2] NVDA — 10-K — 2024-01-01 — Item 7. MD&A"


def test_citation_header_handles_missing_section() -> None:
    c = Citation(
        ordinal=1,
        chunk_id=1,
        filing_id=1,
        ticker="AAPL",
        form="8-K",
        filing_date=date(2023, 5, 1),
        section=None,
        text="t",
    )
    assert "—" in c.header()


def test_critic_report_ok_property() -> None:
    assert CriticReport(unsupported=(), notes="").ok
    bad = CriticReport(
        unsupported=(UnsupportedClaim(claim="x", reason="y"),),
        notes="",
    )
    assert not bad.ok


def test_all_specialists_constants() -> None:
    assert set(ALL_SPECIALISTS) == {"fundamentals", "sentiment", "technical", "risk"}


def test_claim_is_immutable() -> None:
    c = Claim(text="x", citations=(1, 2))
    with pytest.raises(FrozenInstanceError):
        c.text = "y"  # type: ignore[misc]
