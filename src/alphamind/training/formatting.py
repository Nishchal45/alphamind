"""Pure (de)serialisation and chat-message rendering for the fine-tune.

No heavy dependencies here — this is the layer the dataset builder,
trainer, and evaluator all share, and it must import cleanly without
the ``train`` extra. Anything touching torch / transformers lives in
:mod:`alphamind.training.trainer`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from alphamind.agents._json import extract_json_object
from alphamind.llm.base import Message, SystemMessage, UserMessage
from alphamind.training.prompts import EXTRACTION_SYSTEM, build_user_prompt
from alphamind.training.types import Claim


def claims_to_json(claims: Sequence[Claim]) -> str:
    """Serialise claims to the canonical target JSON string.

    This is the exact string the model is trained to produce, so it's
    also the format the evaluator compares against. Compact separators,
    stable key order.
    """
    payload = {"claims": [{"claim": c.text} for c in claims]}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_claims(raw: str) -> list[Claim] | None:
    """Parse a model output (or a teacher target) into claims.

    Returns ``None`` when the payload doesn't parse as the expected
    ``{"claims": [...]}`` schema — the caller decides whether that's a
    dropped training target (builder) or a malformed prediction
    (evaluator). Individual claim entries that aren't ``{"claim": str}``
    with non-empty text are skipped; a well-formed envelope with some
    junk entries still yields the good ones.
    """
    payload = extract_json_object(raw)
    if payload is None:
        return None
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        return None

    claims: list[Claim] = []
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        text = item.get("claim")
        if not isinstance(text, str):
            continue
        cleaned = text.strip()
        if cleaned:
            claims.append(Claim(text=cleaned))
    return claims


def build_extraction_messages(chunk_text: str) -> list[Message]:
    """The system + user turns the extraction task is prompted with.

    Shared by the dataset builder (teacher prompt) and the evaluator
    (candidate prompt) so the model always sees, at inference, exactly
    the framing it was distilled and trained on.
    """
    return [SystemMessage(EXTRACTION_SYSTEM), UserMessage(build_user_prompt(chunk_text))]


__all__ = [
    "build_extraction_messages",
    "claims_to_json",
    "parse_claims",
]
