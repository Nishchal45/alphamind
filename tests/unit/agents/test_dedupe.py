"""Tests for the source-pool dedupe helper."""

from __future__ import annotations

from datetime import date

from alphamind.agents._sources import dedupe_sources
from alphamind.agents.state import Source


def _src(chunk_id: int, ticker: str = "X") -> Source:
    return Source(
        chunk_id=chunk_id,
        filing_id=1,
        ticker=ticker,
        form="10-K",
        filing_date=date(2024, 1, 1),
        section=None,
        text="t",
        score=1.0,
    )


def test_empty_input_returns_empty_list() -> None:
    assert dedupe_sources([]) == []


def test_preserves_first_seen_order() -> None:
    out = dedupe_sources([_src(3), _src(1), _src(2)])
    assert [s.chunk_id for s in out] == [3, 1, 2]


def test_drops_repeated_chunk_ids() -> None:
    out = dedupe_sources([_src(1), _src(2), _src(1), _src(3), _src(2)])
    assert [s.chunk_id for s in out] == [1, 2, 3]


def test_dedupe_compares_only_chunk_id() -> None:
    # Two specialists may pull the same chunk_id with different scores.
    # First seen wins; we don't try to merge metadata.
    a = _src(1, ticker="AAA")
    b = _src(1, ticker="BBB")
    out = dedupe_sources([a, b])
    assert out == [a]
