"""YAML loader for golden-set files.

Golden-set file shape::

    cases:
      - id: aapl-services-2024
        query: "How is AAPL talking about Services growth durability?"
        as_of: 2024-12-31
        expected_topics:
          - "Services"
          - "gross margin"
        required_chunk_ids: [4123, 4218]
        notes: "Covers MD&A excerpt picked manually."

Every field except ``id``, ``query``, and ``as_of`` is optional. The
loader validates types eagerly so the runner doesn't have to defend
against bad inputs at scoring time.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml

from alphamind.eval.types import EvalCase


class GoldenSetError(ValueError):
    """Raised when the golden-set file fails to parse or validate."""


def _coerce_case(raw: dict[str, Any], index: int) -> EvalCase:
    where = f"cases[{index}]"

    try:
        case_id = str(raw["id"])
        query = str(raw["query"])
        as_of_raw = raw["as_of"]
    except KeyError as exc:
        raise GoldenSetError(f"{where}: missing required field {exc.args[0]!r}") from exc

    if isinstance(as_of_raw, date):
        as_of = as_of_raw
    elif isinstance(as_of_raw, str):
        try:
            as_of = date.fromisoformat(as_of_raw)
        except ValueError as exc:
            raise GoldenSetError(f"{where}: as_of must be YYYY-MM-DD, got {as_of_raw!r}") from exc
    else:
        raise GoldenSetError(f"{where}: as_of must be a date or YYYY-MM-DD string")

    topics_raw = raw.get("expected_topics") or []
    if not isinstance(topics_raw, list) or not all(isinstance(t, str) for t in topics_raw):
        raise GoldenSetError(f"{where}: expected_topics must be a list of strings")

    cids_raw = raw.get("required_chunk_ids") or []
    if not isinstance(cids_raw, list) or not all(isinstance(c, int) for c in cids_raw):
        raise GoldenSetError(f"{where}: required_chunk_ids must be a list of integers")

    notes = str(raw.get("notes") or "")

    return EvalCase(
        id=case_id,
        query=query,
        as_of=as_of,
        expected_topics=tuple(topics_raw),
        required_chunk_ids=tuple(cids_raw),
        notes=notes,
    )


def load_golden_set(path: Path) -> list[EvalCase]:
    """Parse a YAML golden-set file into a list of :class:`EvalCase`."""
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise GoldenSetError(f"{path}: top-level value must be a mapping")
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise GoldenSetError(f"{path}: 'cases' must be a non-empty list")

    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise GoldenSetError(f"cases[{i}]: must be a mapping")
        case = _coerce_case(item, i)
        if case.id in seen_ids:
            raise GoldenSetError(f"cases[{i}]: duplicate id {case.id!r}")
        seen_ids.add(case.id)
        cases.append(case)
    return cases


__all__ = ["GoldenSetError", "load_golden_set"]
