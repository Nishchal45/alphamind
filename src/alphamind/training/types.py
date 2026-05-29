"""Typed data shapes for the fine-tune pipeline.

``frozen=True, slots=True`` to match the rest of the codebase
(:mod:`alphamind.agents.state`, :mod:`alphamind.eval.types`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Claim:
    """One extracted claim.

    Intentionally minimal — a single sentence. The chunk it came from
    is tracked on the :class:`TrainingExample`, not here, because the
    fine-tune target is "the claims in this passage" and the source
    binding is the orchestration layer's job (ADR 0011).
    """

    text: str


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One (input chunk, target claims) pair for supervised fine-tuning.

    ``chunk_id`` is carried through so a built dataset can be traced
    back to the source row, and so train/val splitting can be made
    deterministic by id rather than by list position.
    """

    chunk_id: int
    chunk_text: str
    claims: tuple[Claim, ...]

    @property
    def is_empty(self) -> bool:
        """True when the teacher extracted no claims from this chunk.

        Empty examples are still legitimate training signal — "this
        passage contains nothing material" is a thing the model should
        learn to say — but callers sometimes want to filter them, so
        the predicate is surfaced.
        """
        return len(self.claims) == 0


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """A train / validation partition of training examples."""

    train: tuple[TrainingExample, ...]
    validation: tuple[TrainingExample, ...]

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_validation(self) -> int:
        return len(self.validation)


__all__ = ["Claim", "DatasetSplit", "TrainingExample"]
