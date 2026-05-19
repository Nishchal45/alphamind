"""Tolerant JSON extraction for structured LLM output.

The router, specialists, and critic all ask the model to respond with
JSON. Even with a strong "respond with valid JSON" instruction, real
models routinely:

- wrap the JSON in a ```json fenced block,
- prefix it with prose like "Here is the JSON:",
- emit smart quotes or trailing commas.

We could fight this with constrained decoding, but the project's
:class:`alphamind.llm.base.LLMClient` protocol is intentionally narrow —
adding tool-use / constrained-decoding semantics blows the protocol up
just to handle a parsing nicety. The cheaper alternative: a small
extractor that pulls the first ``{...}`` or ``[...]`` block out of the
response, then ``json.loads`` it.

This is enough for the v1 critic and router. When tool use lands later
in Phase 3, those call paths will switch to native structured output
and stop going through here.
"""

from __future__ import annotations

import json
import re
from typing import Any


class StructuredOutputError(ValueError):
    """Raised when an LLM response cannot be parsed as the expected shape."""


# Matches a fenced JSON block: ```json ... ``` or ``` ... ```.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse the first JSON object or array out of ``text``.

    Strategy, in order:

    1. If the response is already valid JSON, use it.
    2. If it has a ```` ```json ```` fence, parse the fenced content.
    3. Otherwise, slice from the first ``{`` or ``[`` to the matching
       closing brace (bracket-balanced; ignores braces inside strings).

    Raises :class:`StructuredOutputError` if none of those work.
    """

    if not text or not text.strip():
        raise StructuredOutputError("empty response")

    # 1. Whole response.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fenced block.
    fence = _FENCE_RE.search(text)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 3. Balanced slice.
    sliced = _balanced_slice(text)
    if sliced is not None:
        try:
            return json.loads(sliced)
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(
                f"found a JSON-like span but it didn't parse: {exc}"
            ) from exc

    raise StructuredOutputError("no JSON object/array found in response")


def _balanced_slice(text: str) -> str | None:
    """Return the smallest balanced ``{...}`` or ``[...]`` block in ``text``.

    String literals are honoured — a ``{`` inside a string doesn't count.
    Returns ``None`` if no balanced span exists.
    """

    open_char = None
    start = -1
    for i, ch in enumerate(text):
        if ch in "{[":
            open_char = ch
            start = i
            break
    if start < 0 or open_char is None:
        return None

    close_char = "}" if open_char == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def require_dict(value: Any, *, context: str) -> dict[str, Any]:
    """Coerce ``value`` to a ``dict[str, Any]`` or raise."""

    if not isinstance(value, dict):
        raise StructuredOutputError(f"{context}: expected JSON object, got {type(value).__name__}")
    # JSON keys are always strings, but be defensive.
    return {str(k): v for k, v in value.items()}


def require_list(value: Any, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise StructuredOutputError(f"{context}: expected JSON array, got {type(value).__name__}")
    return value


def require_str(value: Any, *, context: str) -> str:
    if not isinstance(value, str):
        raise StructuredOutputError(f"{context}: expected string, got {type(value).__name__}")
    return value


__all__ = [
    "StructuredOutputError",
    "extract_json",
    "require_dict",
    "require_list",
    "require_str",
]
