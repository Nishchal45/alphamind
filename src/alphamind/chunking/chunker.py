"""Paragraph-aware text chunker with configurable overlap.

The chunker packs paragraphs greedily up to a character budget, falling back
to sentence- and then word-level splits only for paragraphs that exceed the
budget on their own. Successive chunks share a tail of the previous chunk
so that semantic context straddling a chunk boundary is not lost during
retrieval.

The output type :class:`Chunk` carries enough context (section label,
section title, chunk index) for downstream code to render citations without
re-parsing the source filing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from alphamind.chunking.parser import detect_sections, parse_filing_html

DEFAULT_MAX_CHARS = 4000
DEFAULT_OVERLAP_CHARS = 400

_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])")
_WHITESPACE_RE = re.compile(r"\s+")

_PARAGRAPH_SEPARATOR = "\n\n"


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single retrieval-ready chunk of a parsed filing.

    Attributes
    ----------
    section_label:
        Lowercase section identifier such as ``"item_1a"`` or ``"preamble"``.
    section_title:
        Section heading text if present (e.g. ``"Risk Factors"``).
    chunk_index:
        Zero-based position of this chunk within the filing, in reading order.
    text:
        Chunk body. Paragraphs are joined with ``\\n\\n``.
    char_count:
        ``len(text)``, materialized for convenience.
    """

    section_label: str
    section_title: str | None
    chunk_index: int
    text: str
    char_count: int


def chunk_text(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars`` with overlap.

    Splitting prefers paragraph boundaries, then sentence boundaries, then
    word boundaries — in that order — so the original prose flow is
    preserved whenever the budget allows.

    Parameters
    ----------
    max_chars:
        Hard upper bound on the size of any returned chunk.
    overlap_chars:
        Approximate length of the tail copied from one chunk into the start
        of the next. Must be strictly less than ``max_chars``.
    """

    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if overlap_chars < 0:
        raise ValueError(f"overlap_chars must be non-negative, got {overlap_chars}")
    if overlap_chars >= max_chars:
        raise ValueError(
            f"overlap_chars ({overlap_chars}) must be less than max_chars ({max_chars})"
        )

    units = _atomize(text, max_chars=max_chars)
    if not units:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for unit in units:
        added_len = len(unit) + (len(_PARAGRAPH_SEPARATOR) if buf else 0)
        if buf and buf_len + added_len > max_chars:
            chunks.append(_PARAGRAPH_SEPARATOR.join(buf))
            buf = _tail_units(buf, overlap_chars)
            buf_len = _joined_length(buf)

        added_len = len(unit) + (len(_PARAGRAPH_SEPARATOR) if buf else 0)
        buf.append(unit)
        buf_len += added_len

    if buf:
        chunks.append(_PARAGRAPH_SEPARATOR.join(buf))

    return chunks


def chunk_filing(
    html: bytes | str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Parse a filing's HTML body and return chunks ready for embedding."""

    paragraphs = parse_filing_html(html)
    sections = detect_sections(paragraphs)

    out: list[Chunk] = []
    idx = 0
    for section in sections:
        if not section.paragraphs:
            continue
        body = _PARAGRAPH_SEPARATOR.join(section.paragraphs)
        for piece in chunk_text(body, max_chars=max_chars, overlap_chars=overlap_chars):
            out.append(
                Chunk(
                    section_label=section.label,
                    section_title=section.title,
                    chunk_index=idx,
                    text=piece,
                    char_count=len(piece),
                )
            )
            idx += 1

    return out


def _atomize(text: str, *, max_chars: int) -> list[str]:
    """Break ``text`` into atomic units no larger than ``max_chars`` each.

    A unit is a paragraph if it fits; otherwise it is broken into sentences;
    if a sentence is still too big it is broken into word groups.
    """

    units: list[str] = []
    for raw_paragraph in _PARAGRAPH_RE.split(text):
        paragraph = _WHITESPACE_RE.sub(" ", raw_paragraph).strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue

        for sentence in _split_sentences(paragraph):
            if len(sentence) <= max_chars:
                units.append(sentence)
                continue
            units.extend(_pack_words(sentence, max_chars=max_chars))

    return units


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_RE.split(text)]
    return [s for s in parts if s]


def _pack_words(text: str, *, max_chars: int) -> list[str]:
    """Greedily group words into segments of at most ``max_chars`` chars."""

    out: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in text.split():
        added = len(word) + (1 if current else 0)
        if current and current_len + added > max_chars:
            out.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += added
    if current:
        out.append(" ".join(current))
    return out


def _tail_units(units: list[str], overlap_chars: int) -> list[str]:
    """Return a suffix of ``units`` whose joined length is ~``overlap_chars``.

    The suffix is always shorter than the joined length of all ``units`` —
    we never repeat an entire previous chunk verbatim. If no single unit
    fits in the overlap budget, an empty list is returned and the caller
    starts the next chunk fresh.
    """

    if overlap_chars <= 0 or not units:
        return []

    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        added = len(unit) + (len(_PARAGRAPH_SEPARATOR) if tail else 0)
        if total + added > overlap_chars:
            break
        tail.insert(0, unit)
        total += added

    # Refuse to copy the whole previous chunk into the next one — that would
    # be infinite-loop territory (next emit, same content, repeat).
    if len(tail) == len(units):
        tail = tail[1:]

    return tail


def _joined_length(units: list[str]) -> int:
    if not units:
        return 0
    return sum(len(u) for u in units) + (len(units) - 1) * len(_PARAGRAPH_SEPARATOR)


__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_OVERLAP_CHARS",
    "Chunk",
    "chunk_filing",
    "chunk_text",
]
