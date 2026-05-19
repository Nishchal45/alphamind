"""Tests for :class:`SentimentSpecialist`."""

from __future__ import annotations

import json

import pytest

from alphamind.agents.specialists.sentiment import SentimentSpecialist
from tests.unit.agents.conftest import ScriptedLLMClient
from tests.unit.agents.specialists.conftest import (
    FakeSearch,
    make_chunk,
    make_hit,
    make_state,
    patch_session_scope,
)

pytestmark = pytest.mark.asyncio


async def test_sentiment_specialist_metadata() -> None:
    spec = SentimentSpecialist(
        llm=ScriptedLLMClient([]),
        search=FakeSearch([]),  # type: ignore[arg-type]
    )
    assert spec.name == "sentiment"
    assert spec.domain == "sentiment"
    # Augmentation pulls retrieval toward narrative / outlook language.
    assert "commentary" in spec.query_augmentation.lower()
    assert "outlook" in spec.query_augmentation.lower()
    # Honesty caveat about the missing transcript corpus belongs in the prompt.
    assert "transcript" in spec.system_prompt.lower()


async def test_sentiment_specialist_runs_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_session_scope(
        monkeypatch,
        [
            make_chunk(
                chunk_id=1,
                form="10-Q",
                section="Item 2. MD&A",
                text="Management remains cautiously optimistic about second-half demand.",
            )
        ],
    )
    search = FakeSearch([make_hit(chunk_id=1, section="Item 2. MD&A")])
    llm = ScriptedLLMClient(
        [
            json.dumps(
                {
                    "summary": "Tone is hedged-positive.",
                    "claims": [
                        {
                            "text": "Management uses 'cautiously optimistic' language about H2.",
                            "citations": [1],
                        }
                    ],
                }
            )
        ]
    )

    spec = SentimentSpecialist(llm=llm, search=search)  # type: ignore[arg-type]
    update = await spec(make_state(query="how is management framing the outlook?"))
    report = update["specialist_reports"][0]
    assert report.specialist == "sentiment"
    assert len(report.claims) == 1
    assert "cautiously optimistic" in report.claims[0].text


async def test_sentiment_specialist_augments_retrieval_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_session_scope(monkeypatch, [make_chunk(chunk_id=1)])
    search = FakeSearch([make_hit(chunk_id=1)])
    llm = ScriptedLLMClient([json.dumps({"summary": "s", "claims": []})])

    spec = SentimentSpecialist(llm=llm, search=search)  # type: ignore[arg-type]
    await spec(make_state(query="management commentary on margin"))

    assert search.last_query is not None
    assert "management commentary on margin" in search.last_query
    assert "outlook" in search.last_query
