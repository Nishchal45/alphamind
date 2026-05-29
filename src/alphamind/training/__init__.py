"""Offline fine-tuning pipeline for the claim-extraction SLM.

The package builds a distilled training set (chunk → claims JSON from
the frontier model), fine-tunes a small open model with QLoRA, and
evaluates claim-level precision / recall. See ADR 0011 for the design
and `docs/runbooks/fine-tune.md` for the GPU run procedure.

Layering:

- :mod:`alphamind.training.types` — frozen dataclasses for examples
  and dataset records.
- :mod:`alphamind.training.prompts` — the extraction instruction.
- :mod:`alphamind.training.formatting` — pure example → chat-message
  rendering and claim (de)serialisation.
- :mod:`alphamind.training.dataset` — distillation builder (over the
  ``LLMClient`` Protocol), JSONL IO, deterministic train/val split.
- :mod:`alphamind.training.trainer` — QLoRA config + train loop
  (heavy imports are lazy; needs the ``train`` extra + a GPU).
- :mod:`alphamind.training.evaluate` — claim matching + P/R/F1 + a
  prediction shell that runs over any ``LLMClient``.

Everything except the actual ``train()`` loop and live model
inference is pure and unit-tested without a GPU.
"""

from __future__ import annotations

from alphamind.training.types import (
    Claim,
    DatasetSplit,
    TrainingExample,
)

__all__ = [
    "Claim",
    "DatasetSplit",
    "TrainingExample",
]
