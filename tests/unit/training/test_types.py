"""Tests for the training dataclasses."""

from __future__ import annotations

from alphamind.training.types import Claim, DatasetSplit, TrainingExample


def _example(chunk_id: int, *claims: str) -> TrainingExample:
    return TrainingExample(
        chunk_id=chunk_id,
        chunk_text=f"text-{chunk_id}",
        claims=tuple(Claim(text=c) for c in claims),
    )


def test_training_example_is_empty() -> None:
    assert _example(1).is_empty is True
    assert _example(2, "a claim").is_empty is False


def test_dataset_split_counts() -> None:
    split = DatasetSplit(
        train=(_example(1, "a"), _example(2, "b")),
        validation=(_example(3, "c"),),
    )
    assert split.n_train == 2
    assert split.n_validation == 1


def test_dataclasses_are_frozen() -> None:
    claim = Claim(text="x")
    try:
        claim.text = "y"  # type: ignore[misc]
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Claim should be frozen")
