"""Tests for the eval metric functions.

Each metric is a pure function over a finished :class:`ResearchState`;
the fixtures supply healthy and intentionally-flagged states.
"""

from __future__ import annotations

from alphamind.agents.state import ResearchState, Thesis, ThesisClaim
from alphamind.eval.metrics import (
    chunk_recall,
    citation_coverage,
    citation_validity,
    contradiction_rate,
    hallucination_rate,
    topic_recall,
)


def test_healthy_state_has_full_coverage_and_validity(
    healthy_state: ResearchState,
) -> None:
    assert citation_coverage(healthy_state) == 1.0
    assert citation_validity(healthy_state) == 1.0
    assert hallucination_rate(healthy_state) == 0.0
    assert contradiction_rate(healthy_state) == 0.0


def test_empty_thesis_treats_metrics_as_vacuously_clean() -> None:
    state = ResearchState(
        query="anything",
        thesis=Thesis(summary="(none)", bull_case=(), bear_case=()),
    )
    assert citation_coverage(state) == 1.0
    assert citation_validity(state) == 1.0
    assert hallucination_rate(state) == 0.0
    assert contradiction_rate(state) == 0.0


def test_coverage_drops_when_claims_lack_citations(
    healthy_state: ResearchState,
) -> None:
    bad_thesis = Thesis(
        summary="ok",
        bull_case=(
            ThesisClaim(claim="cited", cited_chunk_ids=(102,)),
            ThesisClaim(claim="uncited", cited_chunk_ids=()),
        ),
        bear_case=(ThesisClaim(claim="cited", cited_chunk_ids=(101,)),),
    )
    state = ResearchState(**healthy_state)
    state["thesis"] = bad_thesis
    # 2 of 3 claims are cited.
    assert citation_coverage(state) == 2 / 3


def test_validity_drops_when_thesis_cites_chunk_not_in_pool(
    healthy_state: ResearchState,
) -> None:
    bad_thesis = Thesis(
        summary="ok",
        bull_case=(ThesisClaim(claim="valid", cited_chunk_ids=(102,)),),
        bear_case=(ThesisClaim(claim="invalid", cited_chunk_ids=(9999,)),),
    )
    state = ResearchState(**healthy_state)
    state["thesis"] = bad_thesis
    # 1 of 2 cites resolves to a real chunk.
    assert citation_validity(state) == 0.5


def test_hallucination_and_contradiction_rates(flagged_state: ResearchState) -> None:
    # Healthy state has 2 claims; flagged state injects 2 issues.
    # Rates use total-claim denominator, so both metrics are 1/2 = 0.5.
    assert hallucination_rate(flagged_state) == 0.5
    assert contradiction_rate(flagged_state) == 0.5


def test_topic_recall_returns_none_when_no_topics(
    healthy_state: ResearchState,
) -> None:
    assert topic_recall(healthy_state, []) is None


def test_topic_recall_counts_case_insensitive_substrings(
    healthy_state: ResearchState,
) -> None:
    # Summary mentions "China" and "margin"; thesis text contains "Gross margin".
    score = topic_recall(healthy_state, ["china", "margin", "nowhere"])
    assert score == 2 / 3


def test_chunk_recall_returns_none_when_no_required(
    healthy_state: ResearchState,
) -> None:
    assert chunk_recall(healthy_state, []) is None


def test_chunk_recall_counts_required_cites(healthy_state: ResearchState) -> None:
    # Healthy thesis cites 101 and 102. We require 101 and 4242 — half hit.
    assert chunk_recall(healthy_state, [101, 4242]) == 0.5
