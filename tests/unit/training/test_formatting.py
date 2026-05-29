"""Tests for claim (de)serialisation and message rendering."""

from __future__ import annotations

from alphamind.training.formatting import (
    build_extraction_messages,
    claims_to_json,
    parse_claims,
)
from alphamind.training.prompts import EXTRACTION_SYSTEM
from alphamind.training.types import Claim


def test_claims_to_json_round_trips_through_parse() -> None:
    claims = [Claim(text="China is 17% of revenue."), Claim(text="Gross margin hit 73%.")]
    raw = claims_to_json(claims)
    parsed = parse_claims(raw)
    assert parsed == claims


def test_claims_to_json_empty_list() -> None:
    assert claims_to_json([]) == '{"claims":[]}'
    assert parse_claims('{"claims":[]}') == []


def test_parse_claims_strips_fences_and_prose() -> None:
    raw = 'Sure! Here you go:\n```json\n{"claims": [{"claim": "Revenue grew 12%."}]}\n```'
    parsed = parse_claims(raw)
    assert parsed == [Claim(text="Revenue grew 12%.")]


def test_parse_claims_skips_malformed_entries_but_keeps_good_ones() -> None:
    raw = '{"claims": [{"claim": "good one"}, "not a dict", {"nope": 1}, {"claim": ""}]}'
    parsed = parse_claims(raw)
    assert parsed == [Claim(text="good one")]


def test_parse_claims_returns_none_on_unparseable() -> None:
    assert parse_claims("definitely not json") is None


def test_parse_claims_returns_none_when_claims_key_missing() -> None:
    assert parse_claims('{"findings": []}') is None


def test_parse_claims_returns_none_when_claims_not_a_list() -> None:
    assert parse_claims('{"claims": "nope"}') is None


def test_parse_claims_trims_whitespace() -> None:
    parsed = parse_claims('{"claims": [{"claim": "  padded  "}]}')
    assert parsed == [Claim(text="padded")]


def test_build_extraction_messages_shape() -> None:
    messages = build_extraction_messages("Some filing text.")
    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[0].content == EXTRACTION_SYSTEM
    assert messages[1].role == "user"
    assert "Some filing text." in messages[1].content
