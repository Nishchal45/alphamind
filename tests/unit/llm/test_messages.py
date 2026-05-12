"""Tests for LLM message construction."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from alphamind.llm.base import (
    AssistantMessage,
    Message,
    SystemMessage,
    UserMessage,
)


def test_message_constructors_emit_correct_roles() -> None:
    assert SystemMessage("hi").role == "system"
    assert UserMessage("hi").role == "user"
    assert AssistantMessage("hi").role == "assistant"


def test_message_rejects_invalid_role() -> None:
    with pytest.raises(ValueError):
        Message(role="root", content="elevate me")  # type: ignore[arg-type]


def test_message_is_frozen() -> None:
    msg = UserMessage("hi")
    with pytest.raises(FrozenInstanceError):
        msg.content = "modified"  # type: ignore[misc]
