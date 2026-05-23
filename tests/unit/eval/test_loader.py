"""Tests for the golden-set YAML loader."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from alphamind.eval.loader import GoldenSetError, load_golden_set


def _write(tmp_path: Path, contents: str) -> Path:
    p = tmp_path / "golden.yaml"
    p.write_text(contents, encoding="utf-8")
    return p


def test_loads_minimal_case(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """\
cases:
  - id: case-1
    query: what is X?
    as_of: 2024-12-31
""",
    )
    [case] = load_golden_set(p)
    assert case.id == "case-1"
    assert case.query == "what is X?"
    assert case.as_of == date(2024, 12, 31)
    assert case.expected_topics == ()
    assert case.required_chunk_ids == ()


def test_loads_full_case(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """\
cases:
  - id: case-1
    query: q
    as_of: 2024-01-01
    expected_topics: [a, b, c]
    required_chunk_ids: [10, 20]
    notes: |
      multi
      line
""",
    )
    [case] = load_golden_set(p)
    assert case.expected_topics == ("a", "b", "c")
    assert case.required_chunk_ids == (10, 20)
    assert "multi" in case.notes


def test_rejects_missing_required_field(tmp_path: Path) -> None:
    p = _write(tmp_path, "cases:\n  - id: x\n    query: q\n")
    with pytest.raises(GoldenSetError, match="as_of"):
        load_golden_set(p)


def test_rejects_bad_as_of_format(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """\
cases:
  - id: case-1
    query: q
    as_of: "yesterday"
""",
    )
    with pytest.raises(GoldenSetError, match="YYYY-MM-DD"):
        load_golden_set(p)


def test_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """\
cases:
  - id: same
    query: q1
    as_of: 2024-01-01
  - id: same
    query: q2
    as_of: 2024-02-01
""",
    )
    with pytest.raises(GoldenSetError, match="duplicate id"):
        load_golden_set(p)


def test_rejects_empty_cases_list(tmp_path: Path) -> None:
    p = _write(tmp_path, "cases: []\n")
    with pytest.raises(GoldenSetError, match="non-empty"):
        load_golden_set(p)


def test_shipped_golden_set_parses() -> None:
    """The golden set checked into the repo must always parse."""
    cases = load_golden_set(Path("evals/golden_set.yaml"))
    assert cases  # at least one case
    ids = {c.id for c in cases}
    assert len(ids) == len(cases)  # no duplicates slipped in
