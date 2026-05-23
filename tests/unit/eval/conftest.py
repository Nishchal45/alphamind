"""Shared fixtures for eval unit tests."""

from __future__ import annotations

from datetime import date

import pytest

from alphamind.agents.state import (
    Critique,
    CritiqueIssue,
    ResearchState,
    Source,
    Thesis,
    ThesisClaim,
    Usage,
)


@pytest.fixture
def healthy_state() -> ResearchState:
    """A finished :class:`ResearchState` where every claim cites a real chunk."""
    return ResearchState(
        query="bull / bear on NVDA?",
        as_of=date(2024, 12, 31),
        top_k=3,
        sources=[
            Source(
                chunk_id=101,
                filing_id=1,
                ticker="NVDA",
                form="10-K",
                filing_date=date(2024, 2, 1),
                section="Item 1A",
                text="Sales to customers in China represented 17% of total revenue.",
                score=0.91,
            ),
            Source(
                chunk_id=102,
                filing_id=1,
                ticker="NVDA",
                form="10-K",
                filing_date=date(2024, 2, 1),
                section="Item 7",
                text="Gross margin expanded to 73%.",
                score=0.85,
            ),
        ],
        thesis=Thesis(
            summary=(
                "Margins are strong; geopolitical risk is the dominant downside for NVDA in China."
            ),
            bull_case=(ThesisClaim(claim="Gross margin expanded.", cited_chunk_ids=(102,)),),
            bear_case=(ThesisClaim(claim="China revenue concentration.", cited_chunk_ids=(101,)),),
        ),
        critique=Critique(issues=(), parse_error=None),
        usage=[
            Usage(node="router", model="stub", input_tokens=10, output_tokens=5),
            Usage(node="fundamentals", model="stub", input_tokens=40, output_tokens=20),
            Usage(node="synthesizer", model="stub", input_tokens=20, output_tokens=10),
            Usage(node="critic", model="stub", input_tokens=30, output_tokens=8),
        ],
    )


@pytest.fixture
def flagged_state(healthy_state: ResearchState) -> ResearchState:
    """Healthy state with two critic issues added."""
    new = ResearchState(**healthy_state)
    new["critique"] = Critique(
        issues=(
            CritiqueIssue(
                kind="unsupported",
                claim="Gross margin expanded.",
                detail=(
                    "Chunk 102 mentions gross margin but the bull claim implies operating margin."
                ),
                cited_chunk_ids=(102,),
            ),
            CritiqueIssue(
                kind="contradiction",
                claim="China revenue concentration.",
                detail="Conflicts with the bull side framing.",
                cited_chunk_ids=(101,),
            ),
        ),
        parse_error=None,
    )
    return new
