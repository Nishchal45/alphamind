"""Tests for the technical specialist's no-data stub behaviour.

The technical specialist is shipped as a stub until the market-data
adapter exists. The contract these tests pin down:

- It satisfies :class:`SpecialistBase` and the graph constructor
  signature (no graph-side special case needed).
- It never calls the LLM or the search backend.
- It returns an empty :class:`SpecialistReport` with the documented
  no-data reason in the summary.
"""

from __future__ import annotations

import pytest

from alphamind.agents.specialists.technical import NO_DATA_REASON, TechnicalSpecialist
from tests.unit.agents.conftest import ScriptedLLMClient
from tests.unit.agents.specialists.conftest import FakeSearch, make_state

pytestmark = pytest.mark.asyncio


async def test_technical_specialist_metadata() -> None:
    spec = TechnicalSpecialist(
        llm=ScriptedLLMClient([]),
        search=FakeSearch([]),  # type: ignore[arg-type]
    )
    assert spec.name == "technical"
    assert spec.domain == "technical"


async def test_technical_specialist_returns_empty_report_without_calling_llm() -> None:
    # An LLM with zero scripted responses will raise if called, which is
    # exactly what we want — the stub must not reach the model.
    llm = ScriptedLLMClient([])
    search = FakeSearch([])  # likewise: no scripted hits

    spec = TechnicalSpecialist(llm=llm, search=search)  # type: ignore[arg-type]
    update = await spec(make_state(query="momentum on NVDA"))

    report = update["specialist_reports"][0]
    assert report.specialist == "technical"
    assert report.claims == ()
    assert report.citations == ()
    assert NO_DATA_REASON in report.summary

    # The LLM and search backends were never touched.
    assert llm.calls == []
    assert search.last_query is None
