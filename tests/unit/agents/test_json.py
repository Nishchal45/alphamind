"""Tests for the agent JSON extractor."""

from __future__ import annotations

from alphamind.agents._json import extract_json_object


def test_parses_clean_object() -> None:
    parsed = extract_json_object('{"a": 1, "b": [2, 3]}')
    assert parsed == {"a": 1, "b": [2, 3]}


def test_strips_fence() -> None:
    raw = 'Sure, here you go:\n```json\n{"primary": "fundamentals"}\n```\n'
    assert extract_json_object(raw) == {"primary": "fundamentals"}


def test_recovers_from_prefix_and_suffix() -> None:
    raw = 'Sure! {\n  "x": 1\n} hope that helps.'
    assert extract_json_object(raw) == {"x": 1}


def test_returns_none_on_garbage() -> None:
    assert extract_json_object("not json at all") is None


def test_rejects_non_object_root() -> None:
    # Arrays and scalars aren't acceptable — every contract returns {}.
    assert extract_json_object("[1, 2, 3]") is None
    assert extract_json_object("42") is None
