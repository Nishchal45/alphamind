"""vLLM-backed local model adapter implementing :class:`LLMClient`.

Serves a Hugging Face base model, optionally with a LoRA adapter (the
artefact ADR 0011 produces), behind the same Protocol the rest of the
agent layer programs against. See ADR 0012.

vLLM is a heavy, CUDA/Linux-centric dependency behind the optional
``serve-local`` extra. It is imported lazily — the model is built on
first ``complete()``, not at import or construction — so this module
imports cleanly without the extra, and a clean :class:`LLMClientError`
is raised if generation is attempted without it.

The message-rendering logic is pure and unit-tested; the model
construction and generation are isolated behind the lazy import and
run in a worker thread (vLLM's ``generate`` is synchronous) so the
event loop stays free, mirroring :class:`CrossEncoderReranker`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from alphamind.llm.base import LLMClient, LLMClientError, LLMResponse, Message

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def to_chat_messages(
    messages: Sequence[Message],
    *,
    system: str | None = None,
) -> list[dict[str, str]]:
    """Flatten the Protocol messages (+ optional ``system``) to chat dicts.

    A standalone ``system`` argument is prepended as a system turn; any
    system-role messages already in ``messages`` are kept in place.
    Pure — this is the unit-tested half of the backend, separate from
    the tokenizer/vLLM calls that need a GPU.
    """
    chat: list[dict[str, str]] = []
    if system:
        chat.append({"role": "system", "content": system})
    for msg in messages:
        chat.append({"role": msg.role, "content": msg.content})
    return chat


class LocalLLMClient(LLMClient):
    """vLLM-backed implementation of :class:`LLMClient`."""

    def __init__(
        self,
        *,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_path: str | None = None,
        default_model: str | None = None,
        max_model_len: int = 4096,
    ) -> None:
        self._base_model = base_model
        self._adapter_path = adapter_path
        # The reported model id: the adapter if present, else the base.
        self._default_model = default_model or adapter_path or base_model
        self._max_model_len = max_model_len
        self._engine: Any | None = None
        self._tokenizer: Any | None = None
        self._lora_request: Any | None = None

    @property
    def default_model(self) -> str:
        return self._default_model

    def _ensure_loaded(self) -> None:  # pragma: no cover - GPU-only path
        """Lazily build the vLLM engine + tokenizer on first use."""
        if self._engine is not None:
            return
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
            from vllm import LLM  # noqa: PLC0415
            from vllm.lora.request import LoRARequest  # noqa: PLC0415
        except ImportError as exc:
            raise LLMClientError(
                "the 'serve-local' extra (vllm) is required for llm_backend='local': "
                "install it with `uv sync --extra serve-local` (needs a CUDA GPU)"
            ) from exc

        logger.info("loading vLLM engine: base=%s adapter=%s", self._base_model, self._adapter_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self._base_model)
        self._engine = LLM(
            model=self._base_model,
            enable_lora=self._adapter_path is not None,
            max_model_len=self._max_model_len,
        )
        if self._adapter_path is not None:
            self._lora_request = LoRARequest("claim-extractor", 1, self._adapter_path)

    def _generate(  # pragma: no cover - GPU-only path
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, int]:
        """Synchronous generate. Runs in a worker thread via ``complete``."""
        from vllm import SamplingParams  # noqa: PLC0415

        assert self._engine is not None  # _ensure_loaded ran
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self._engine.generate(
            [prompt],
            params,
            lora_request=self._lora_request,
        )
        result = outputs[0]
        text = result.outputs[0].text
        input_tokens = len(result.prompt_token_ids)
        output_tokens = len(result.outputs[0].token_ids)
        return text, input_tokens, output_tokens

    def _render_prompt(self, messages: Sequence[Message], system: str | None) -> str:
        """Apply the base model's chat template to the messages.

        Needs the tokenizer (loaded), so it's part of the GPU path.
        """
        assert self._tokenizer is not None  # pragma: no cover - GPU-only path
        chat = to_chat_messages(messages, system=system)
        rendered: str = self._tokenizer.apply_chat_template(  # pragma: no cover
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered

    async def complete(  # pragma: no cover - GPU-only path
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        if not messages:
            raise ValueError("messages must be non-empty")

        self._ensure_loaded()
        prompt = self._render_prompt(messages, system)

        try:
            text, input_tokens, output_tokens = await asyncio.to_thread(
                self._generate,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            raise LLMClientError(f"local vLLM generation failed: {exc}") from exc

        return LLMResponse(
            content=text,
            model=model or self._default_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason="stop",
        )


__all__ = ["DEFAULT_BASE_MODEL", "LocalLLMClient", "to_chat_messages"]
