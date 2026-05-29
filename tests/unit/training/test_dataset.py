"""Tests for the distillation dataset builder, JSONL IO, and split."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from alphamind.llm.base import LLMClientError, LLMResponse, Message
from alphamind.training.dataset import (
    ChunkInput,
    build_examples,
    read_jsonl,
    split_examples,
    write_jsonl,
)
from alphamind.training.types import Claim, TrainingExample


@dataclass
class KeyedTeacher:
    """Teacher stub: returns a canned response keyed by chunk text substring.

    Lets a build over several chunks return a different (or malformed,
    or erroring) target per chunk regardless of call order.
    """

    by_marker: dict[str, str] = field(default_factory=dict)
    raise_for: set[str] = field(default_factory=set)
    default_model: str = "teacher-stub"
    calls: list[str] = field(default_factory=list)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        user = messages[-1].content
        self.calls.append(user)
        for marker in self.raise_for:
            if marker in user:
                raise LLMClientError(f"teacher boom on {marker!r}")
        content = "{}"
        for marker, response in self.by_marker.items():
            if marker in user:
                content = response
                break
        return LLMResponse(
            content=content,
            model=model or self.default_model,
            input_tokens=10,
            output_tokens=10,
            stop_reason="end_turn",
        )


async def test_build_examples_happy_path() -> None:
    chunks = [
        ChunkInput(chunk_id=1, text="ALPHA passage about revenue"),
        ChunkInput(chunk_id=2, text="BETA passage about risk"),
    ]
    teacher = KeyedTeacher(
        by_marker={
            "ALPHA": '{"claims": [{"claim": "Revenue grew 12%."}]}',
            "BETA": '{"claims": [{"claim": "Litigation is pending."}]}',
        }
    )
    examples, stats = await build_examples(chunks, llm=teacher)

    assert stats.seen == 2
    assert stats.written == 2
    assert stats.dropped_unparseable == 0
    assert stats.dropped_errored == 0
    assert examples[0].chunk_id == 1
    assert examples[0].claims == (Claim(text="Revenue grew 12%."),)
    assert examples[1].claims == (Claim(text="Litigation is pending."),)


async def test_build_examples_drops_unparseable_teacher_output() -> None:
    chunks = [
        ChunkInput(chunk_id=1, text="GOOD passage"),
        ChunkInput(chunk_id=2, text="BAD passage"),
    ]
    teacher = KeyedTeacher(
        by_marker={
            "GOOD": '{"claims": [{"claim": "ok"}]}',
            "BAD": "the teacher rambled instead of returning json",
        }
    )
    examples, stats = await build_examples(chunks, llm=teacher)
    assert stats.written == 1
    assert stats.dropped_unparseable == 1
    assert [e.chunk_id for e in examples] == [1]


async def test_build_examples_isolates_teacher_errors() -> None:
    chunks = [
        ChunkInput(chunk_id=1, text="OK passage"),
        ChunkInput(chunk_id=2, text="EXPLODE passage"),
    ]
    teacher = KeyedTeacher(
        by_marker={"OK": '{"claims": [{"claim": "ok"}]}'},
        raise_for={"EXPLODE"},
    )
    examples, stats = await build_examples(chunks, llm=teacher)
    assert stats.written == 1
    assert stats.dropped_errored == 1
    assert [e.chunk_id for e in examples] == [1]


async def test_build_examples_keeps_empty_by_default() -> None:
    chunks = [ChunkInput(chunk_id=1, text="EMPTY passage")]
    teacher = KeyedTeacher(by_marker={"EMPTY": '{"claims": []}'})
    examples, stats = await build_examples(chunks, llm=teacher)
    assert stats.written == 1
    assert examples[0].is_empty


async def test_build_examples_can_drop_empty() -> None:
    chunks = [ChunkInput(chunk_id=1, text="EMPTY passage")]
    teacher = KeyedTeacher(by_marker={"EMPTY": '{"claims": []}'})
    examples, stats = await build_examples(chunks, llm=teacher, keep_empty=False)
    assert stats.written == 0
    assert examples == []


def test_jsonl_round_trip(tmp_path: Path) -> None:
    examples = [
        TrainingExample(
            chunk_id=1,
            chunk_text="passage one",
            claims=(Claim(text="claim a"), Claim(text="claim b")),
        ),
        TrainingExample(chunk_id=2, chunk_text="passage two", claims=()),
    ]
    path = tmp_path / "data.jsonl"
    n = write_jsonl(examples, path)
    assert n == 2

    loaded = read_jsonl(path)
    assert loaded == examples


def test_read_jsonl_skips_corrupt_target(tmp_path: Path) -> None:
    good = json.dumps(
        {"chunk_id": 1, "chunk_text": "ok", "target": '{"claims":[{"claim":"good"}]}'}
    )
    bad = json.dumps({"chunk_id": 2, "chunk_text": "bad", "target": "not json"})
    path = tmp_path / "data.jsonl"
    path.write_text(f"{good}\n{bad}\n", encoding="utf-8")
    loaded = read_jsonl(path)
    assert [e.chunk_id for e in loaded] == [1]


def _examples(n: int) -> list[TrainingExample]:
    return [
        TrainingExample(chunk_id=i, chunk_text=f"t{i}", claims=(Claim(text="c"),))
        for i in range(n)
    ]


def test_split_is_deterministic_by_seed() -> None:
    examples = _examples(200)
    a = split_examples(examples, validation_fraction=0.2, seed=42)
    b = split_examples(examples, validation_fraction=0.2, seed=42)
    assert [e.chunk_id for e in a.validation] == [e.chunk_id for e in b.validation]


def test_split_is_stable_when_examples_are_added() -> None:
    # A chunk's partition must not change when new chunks are added.
    small = split_examples(_examples(100), validation_fraction=0.2, seed=7)
    large = split_examples(_examples(200), validation_fraction=0.2, seed=7)
    small_val_ids = {e.chunk_id for e in small.validation}
    large_val_ids = {e.chunk_id for e in large.validation}
    # Every id that was in validation in the small set is still there.
    assert small_val_ids <= large_val_ids


def test_split_fraction_is_approximately_honoured() -> None:
    split = split_examples(_examples(1000), validation_fraction=0.2, seed=1)
    # Hash-based split won't be exact; assert it's in a sane band.
    assert 150 <= split.n_validation <= 250
    assert split.n_train + split.n_validation == 1000


def test_split_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        split_examples(_examples(10), validation_fraction=1.0)
