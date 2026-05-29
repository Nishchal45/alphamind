"""QLoRA fine-tune configuration + train loop.

The config assembly and the example-to-chat rendering are pure and
tested. The ``train()`` loop lazily imports torch / transformers /
peft / trl (the ``train`` extra) and needs a GPU; it's a thin wrapper
that turns a :class:`alphamind.training.types.DatasetSplit` into a
saved LoRA adapter. See ADR 0011 and ``docs/runbooks/fine-tune.md``.

Defaults match ADR 0011: Qwen2.5-1.5B-Instruct base, 4-bit NF4 QLoRA,
rank 16 / alpha 32, attention + MLP target modules.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alphamind.training.formatting import claims_to_json
from alphamind.training.prompts import EXTRACTION_SYSTEM, build_user_prompt
from alphamind.training.types import DatasetSplit, TrainingExample

logger = logging.getLogger(__name__)


class TrainingDependencyError(RuntimeError):
    """Raised when the ``train`` extra (torch/peft/trl) isn't installed."""


_DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Hyperparameters for the QLoRA fine-tune.

    Every field has an ADR-0011 default; the CLI / runbook overrides
    without code changes.
    """

    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: Path = Path("checkpoints/claim-extractor")

    # LoRA.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = _DEFAULT_TARGET_MODULES

    # Quantisation.
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Optimisation.
    epochs: float = 1.0
    learning_rate: float = 2e-4
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    seed: int = 0

    def __post_init__(self) -> None:
        if self.lora_r <= 0:
            raise ValueError("lora_r must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not self.target_modules:
            raise ValueError("target_modules must be non-empty")


def example_to_chat(example: TrainingExample) -> list[dict[str, str]]:
    """Render one example as a chat-message list for SFT.

    The system + user turns match what the model sees at inference
    (so training and serving framing never drift); the assistant turn
    is the canonical claims-JSON target. ``train()`` applies the
    tokenizer's chat template to this list.
    """
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": build_user_prompt(example.chunk_text)},
        {"role": "assistant", "content": claims_to_json(example.claims)},
    ]


def build_lora_kwargs(cfg: TrainConfig) -> dict[str, Any]:
    """Assemble the kwargs for ``peft.LoraConfig``. Pure / testable."""
    return {
        "r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "target_modules": list(cfg.target_modules),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def build_quant_kwargs(cfg: TrainConfig) -> dict[str, Any] | None:
    """Assemble the kwargs for ``transformers.BitsAndBytesConfig``.

    Returns ``None`` when 4-bit loading is disabled, so the caller
    skips quantisation entirely (full-precision base). Pure / testable.
    """
    if not cfg.load_in_4bit:
        return None
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": cfg.bnb_4bit_quant_type,
        "bnb_4bit_use_double_quant": cfg.bnb_4bit_use_double_quant,
        # bnb_4bit_compute_dtype is set in train() where torch is imported.
    }


def _require_train_extra() -> None:
    """Raise a clear error if the heavy training stack isn't importable."""
    try:
        import peft  # noqa: F401, PLC0415
        import torch  # noqa: F401, PLC0415
        import transformers  # noqa: F401, PLC0415
        import trl  # noqa: F401, PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise TrainingDependencyError(
            "the 'train' extra is required for fine-tuning: "
            "install it with `uv sync --extra train` (needs a CUDA GPU)"
        ) from exc


def train(split: DatasetSplit, cfg: TrainConfig | None = None) -> Path:  # pragma: no cover
    """Run the QLoRA fine-tune and return the saved-adapter path.

    GPU-only. Not exercised by the unit suite — see
    ``docs/runbooks/fine-tune.md`` for the run procedure. The pure
    pieces it composes (``example_to_chat``, ``build_lora_kwargs``,
    ``build_quant_kwargs``) are tested independently.
    """
    cfg = cfg or TrainConfig()
    _require_train_extra()

    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from peft import LoraConfig  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    logger.info("loading tokenizer + base model: %s", cfg.base_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_kwargs = build_quant_kwargs(cfg)
    quant_config = None
    if quant_kwargs is not None:
        quant_config = BitsAndBytesConfig(
            bnb_4bit_compute_dtype=torch.bfloat16,
            **quant_kwargs,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    def _render(examples: Sequence[TrainingExample]) -> Dataset:
        rows = [
            {
                "text": tokenizer.apply_chat_template(
                    example_to_chat(ex),
                    tokenize=False,
                    add_generation_prompt=False,
                )
            }
            for ex in examples
        ]
        return Dataset.from_list(rows)

    train_ds = _render(split.train)
    eval_ds = _render(split.validation) if split.validation else None

    lora_config = LoraConfig(**build_lora_kwargs(cfg))

    sft_config = SFTConfig(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_seq_length=cfg.max_seq_length,
        warmup_ratio=cfg.warmup_ratio,
        seed=cfg.seed,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    logger.info("starting fine-tune: %d train / %d val", split.n_train, split.n_validation)
    trainer.train()

    adapter_path = cfg.output_dir / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("saved adapter to %s", adapter_path)
    return adapter_path


__all__ = [
    "TrainConfig",
    "TrainingDependencyError",
    "build_lora_kwargs",
    "build_quant_kwargs",
    "example_to_chat",
    "train",
]
