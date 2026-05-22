"""JSON parsing helpers for agent nodes.

Models occasionally wrap their JSON in ``` fences, prefix a sentence,
or trail a sign-off, even when instructed not to. The helpers here
recover the JSON payload from those variations and parse it; on a hard
parse failure they return ``None`` so the caller can decide whether to
treat it as a node-level failure or a recoverable degradation.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Return the first JSON object found in ``raw``, or ``None``.

    Tries, in order: a fenced ``` block, a raw ``{ ... }`` slice, and
    finally the whole string. Only ``dict`` payloads are returned;
    arrays and scalars are rejected (every agent contract here outputs
    a top-level object).
    """
    candidates = []

    fenced = _FENCE_RE.search(raw)
    if fenced is not None:
        candidates.append(fenced.group(1))

    # Slice between the first '{' and the matching last '}'. Cheaper
    # than a real parser, good enough for one nested level of objects.
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start : end + 1])

    candidates.append(raw)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None
