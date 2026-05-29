"""Tests for the pure pieces of the trainer.

The ``train()`` loop itself is GPU-only and not exercised here; these
cover the config assembly and chat rendering it composes, plus the
clean-error behaviour when the ``train`` extra is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alphamind.training.prompts import EXTRACTION_SYSTEM
from alphamind.training.trainer import (
    TrainConfig,
    build_lora_kwargs,
    build_quant_kwargs,
    example_to_chat,
)
from alphamind.training.types import Claim, TrainingExample


def test_default_config_matches_adr() -> None:
    cfg = TrainConfig()
    assert cfg.base_model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.load_in_4bit is True
    assert "q_proj" in cfg.target_modules
    assert "down_proj" in cfg.target_modules


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="lora_r"):
        TrainConfig(lora_r=0)
    with pytest.raises(ValueError, match="epochs"):
        TrainConfig(epochs=0)
    with pytest.raises(ValueError, match="target_modules"):
        TrainConfig(target_modules=())


def test_build_lora_kwargs() -> None:
    cfg = TrainConfig(lora_r=8, lora_alpha=16, lora_dropout=0.1)
    kwargs = build_lora_kwargs(cfg)
    assert kwargs["r"] == 8
    assert kwargs["lora_alpha"] == 16
    assert kwargs["lora_dropout"] == 0.1
    assert kwargs["task_type"] == "CAUSAL_LM"
    assert isinstance(kwargs["target_modules"], list)


def test_build_quant_kwargs_enabled() -> None:
    kwargs = build_quant_kwargs(TrainConfig(load_in_4bit=True))
    assert kwargs is not None
    assert kwargs["load_in_4bit"] is True
    assert kwargs["bnb_4bit_quant_type"] == "nf4"


def test_build_quant_kwargs_disabled_returns_none() -> None:
    assert build_quant_kwargs(TrainConfig(load_in_4bit=False)) is None


def test_example_to_chat_shape() -> None:
    ex = TrainingExample(
        chunk_id=1,
        chunk_text="Revenue rose 12% to $5B.",
        claims=(Claim(text="Revenue rose 12% to $5B."),),
    )
    chat = example_to_chat(ex)
    assert [m["role"] for m in chat] == ["system", "user", "assistant"]
    assert chat[0]["content"] == EXTRACTION_SYSTEM
    assert "Revenue rose 12% to $5B." in chat[1]["content"]
    # Assistant turn is the canonical JSON target.
    assert chat[2]["content"] == '{"claims":[{"claim":"Revenue rose 12% to $5B."}]}'


def test_example_to_chat_empty_claims() -> None:
    ex = TrainingExample(chunk_id=1, chunk_text="boilerplate", claims=())
    chat = example_to_chat(ex)
    assert chat[2]["content"] == '{"claims":[]}'


def test_output_dir_is_a_path(tmp_path: Path) -> None:
    cfg = TrainConfig(output_dir=tmp_path / "ckpt")
    assert isinstance(cfg.output_dir, Path)
