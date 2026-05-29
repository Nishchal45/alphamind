"""Tests for the vLLM local backend.

Only the pure pieces are exercised: message rendering, construction
without a model load, the default-model logic, and the factory wiring.
The GPU path (engine build + generate) is not unit-tested — it needs
CUDA and weights; see docs/runbooks/fine-tune.md.
"""

from __future__ import annotations

from alphamind.llm.base import SystemMessage, UserMessage
from alphamind.llm.local import DEFAULT_BASE_MODEL, LocalLLMClient, to_chat_messages


def test_to_chat_messages_prepends_standalone_system() -> None:
    chat = to_chat_messages([UserMessage("hi")], system="be terse")
    assert chat == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


def test_to_chat_messages_keeps_inline_system() -> None:
    chat = to_chat_messages([SystemMessage("inline"), UserMessage("hi")])
    assert chat == [
        {"role": "system", "content": "inline"},
        {"role": "user", "content": "hi"},
    ]


def test_to_chat_messages_no_system() -> None:
    chat = to_chat_messages([UserMessage("just this")])
    assert chat == [{"role": "user", "content": "just this"}]


def test_construction_does_not_load_model() -> None:
    # Constructing must be cheap — no vLLM import, no weights.
    client = LocalLLMClient()
    assert client.default_model == DEFAULT_BASE_MODEL


def test_default_model_prefers_adapter_path() -> None:
    client = LocalLLMClient(base_model="Qwen/Qwen2.5-1.5B-Instruct", adapter_path="/ckpt/adapter")
    assert client.default_model == "/ckpt/adapter"


def test_default_model_explicit_override() -> None:
    client = LocalLLMClient(adapter_path="/ckpt/adapter", default_model="my-model-v1")
    assert client.default_model == "my-model-v1"


def test_default_model_falls_back_to_base() -> None:
    client = LocalLLMClient(base_model="some/base")
    assert client.default_model == "some/base"
