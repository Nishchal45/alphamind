"""Tests for the token-aware splitter."""

from __future__ import annotations

import pytest

from alphamind.retrieval.chunking.splitter import TokenAwareSplitter, count_tokens


def test_count_tokens_matches_known_lengths() -> None:
    # Empty -> 0; non-empty -> positive.
    assert count_tokens("") == 0
    assert count_tokens("hello") > 0


def test_split_returns_non_empty_spans_with_overlap() -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 200
    splitter = TokenAwareSplitter(target_tokens=64, overlap_tokens=16)

    spans = list(splitter.split(text))

    assert len(spans) >= 2
    for char_start, char_end, token_count in spans:
        assert char_end > char_start
        assert 1 <= token_count <= 64

    # Adjacent spans overlap in character space.
    for prev, curr in zip(spans, spans[1:], strict=False):
        assert curr[0] < prev[1], "spans must overlap"


def test_split_covers_the_full_document() -> None:
    text = "alpha beta gamma delta epsilon. " * 50
    splitter = TokenAwareSplitter(target_tokens=32, overlap_tokens=8)

    spans = list(splitter.split(text))

    # Last span ends at document length.
    assert spans[-1][1] == len(text)


def test_short_text_yields_one_span() -> None:
    text = "Just a short sentence."
    splitter = TokenAwareSplitter(target_tokens=512, overlap_tokens=64)

    spans = list(splitter.split(text))

    assert len(spans) == 1
    assert spans[0][0] == 0
    assert spans[0][1] == len(text)


def test_invalid_arguments_rejected() -> None:
    with pytest.raises(ValueError):
        TokenAwareSplitter(target_tokens=0, overlap_tokens=0)
    with pytest.raises(ValueError):
        TokenAwareSplitter(target_tokens=64, overlap_tokens=-1)
    with pytest.raises(ValueError):
        TokenAwareSplitter(target_tokens=64, overlap_tokens=64)


def test_empty_input_yields_no_spans() -> None:
    splitter = TokenAwareSplitter(target_tokens=64, overlap_tokens=8)

    assert list(splitter.split("")) == []
