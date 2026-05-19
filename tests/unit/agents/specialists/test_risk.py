"""Tests for :class:`RiskSpecialist`.

The deep base-pipeline tests live in
:mod:`tests.unit.agents.specialists.test_fundamentals`. This file
covers the risk-specific surface: name, augmentation, prompt content,
and an end-to-end smoke test.
"""

from __future__ import annotations

import json

import pytest

from alphamind.agents.specialists.risk import RiskSpecialist
from tests.unit.agents.conftest import ScriptedLLMClient
from tests.unit.agents.specialists.conftest import (
    FakeSearch,
    make_chunk,
    make_hit,
    make_state,
    patch_session_scope,
)

pytestmark = pytest.mark.asyncio


async def test_risk_specialist_metadata() -> None:
    spec = RiskSpecialist(
        llm=ScriptedLLMClient([]),
        search=FakeSearch([]),  # type: ignore[arg-type]
    )
    assert spec.name == "risk"
    assert spec.domain == "risk"
    # Augmentation pulls retrieval toward Item 1A and legal proceedings.
    assert "risk factor" in spec.query_augmentation.lower()
    assert "litigation" in spec.query_augmentation.lower()
    # Sanity: the system prompt names Item 1A so the LLM has the right frame.
    assert "Item 1A" in spec.system_prompt


async def test_risk_specialist_runs_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_session_scope(
        monkeypatch,
        [
            make_chunk(
                chunk_id=1,
                form="10-K",
                section="Item 1A. Risk Factors",
                text="The Company is subject to a pending EU regulatory investigation.",
            )
        ],
    )
    search = FakeSearch([make_hit(chunk_id=1, section="Item 1A. Risk Factors")])
    llm = ScriptedLLMClient(
        [
            json.dumps(
                {
                    "summary": "EU antitrust investigation is the most material risk.",
                    "claims": [
                        {"text": "Pending EU regulatory investigation.", "citations": [1]},
                    ],
                }
            )
        ]
    )

    spec = RiskSpecialist(llm=llm, search=search)  # type: ignore[arg-type]
    update = await spec(make_state(query="what are the regulatory risks to NVDA?"))
    report = update["specialist_reports"][0]
    assert report.specialist == "risk"
    assert len(report.claims) == 1
    assert "EU" in report.claims[0].text


async def test_risk_specialist_augments_retrieval_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_session_scope(monkeypatch, [make_chunk(chunk_id=1)])
    search = FakeSearch([make_hit(chunk_id=1)])
    llm = ScriptedLLMClient([json.dumps({"summary": "s", "claims": []})])

    spec = RiskSpecialist(llm=llm, search=search)  # type: ignore[arg-type]
    await spec(make_state(query="regulatory risks"))

    assert search.last_query is not None
    assert "regulatory risks" in search.last_query
    assert "litigation" in search.last_query
