"""Tests for the tolerant JSON extractor."""

from __future__ import annotations

import pytest

from alphamind.agents.json_utils import (
    StructuredOutputError,
    extract_json,
    require_dict,
    require_list,
    require_str,
)


def test_parses_plain_json_object() -> None:
    assert extract_json('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_parses_plain_json_array() -> None:
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_strips_fenced_json_block() -> None:
    response = """\
Here is the JSON you asked for:

```json
{"specialists": ["fundamentals"], "rationale": "revenue question"}
```
"""
    assert extract_json(response) == {
        "specialists": ["fundamentals"],
        "rationale": "revenue question",
    }


def test_strips_unlabelled_fence() -> None:
    response = "Sure:\n```\n[1, 2]\n```\nLet me know if you need more."
    assert extract_json(response) == [1, 2]


def test_balanced_slice_handles_prose_prefix() -> None:
    response = 'Output: {"hello": "world"} and then some trailing nonsense'
    assert extract_json(response) == {"hello": "world"}


def test_balanced_slice_ignores_braces_in_strings() -> None:
    response = 'Result: {"sentence": "with a { in it"} done.'
    assert extract_json(response) == {"sentence": "with a { in it"}


def test_raises_when_no_json_present() -> None:
    with pytest.raises(StructuredOutputError):
        extract_json("just prose, no structure at all")


def test_raises_when_empty() -> None:
    with pytest.raises(StructuredOutputError):
        extract_json("   ")


def test_raises_when_span_is_malformed() -> None:
    # A balanced-looking but malformed JSON span should raise rather than
    # silently returning prose.
    with pytest.raises(StructuredOutputError):
        extract_json("garbage {invalid json here}")


def test_require_dict_rejects_array() -> None:
    with pytest.raises(StructuredOutputError):
        require_dict([1, 2], context="x")


def test_require_list_rejects_dict() -> None:
    with pytest.raises(StructuredOutputError):
        require_list({"k": "v"}, context="x")


def test_require_str_rejects_number() -> None:
    with pytest.raises(StructuredOutputError):
        require_str(42, context="x")
