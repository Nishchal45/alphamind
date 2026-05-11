"""Parse SEC filing HTML into structured text suitable for chunking.

The parser does two jobs:

1. Strip HTML/inline-XBRL down to a flat list of text paragraphs, preserving
   reading order and collapsing whitespace.
2. Group those paragraphs into sections keyed on the filing's own structure
   (``Item 1A``, ``Item 7``, ``Part II``, etc.).

Section detection uses a heading-pattern heuristic rather than the DOM:
SEC filings are inconsistent about whether section titles live in ``<h*>``
tags or are styled inline with ``<b>``/``<font>``, but the textual form
``Item <n>`` / ``Part <roman>`` is reliable across filers and decades.

Filings typically contain a table of contents that repeats every section
heading. The deduplication pass at the end of :func:`detect_sections`
keeps only the *longest* section per label, so the TOC entries (small)
are discarded in favour of the body (large).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

# A "heading-ish" block must be short. Anything longer is body text that
# happens to start with the words "Item N" (e.g. a sentence in a paragraph).
_MAX_HEADING_CHARS = 200

# Section headings in SEC filings routinely use en-dash or em-dash to
# separate the item label from the title (e.g. "Item 1A — Risk Factors").
# Match both alongside ASCII hyphen and colon.
_ITEM_RE = re.compile(
    r"^\s*Item\s+(\d+(?:\.\d+)?[A-Za-z]?)\b\.?\s*[:\-–—]?\s*(.*)$",  # noqa: RUF001
    re.IGNORECASE,
)
_PART_RE = re.compile(
    r"^\s*Part\s+([IVX]+)\b\.?\s*[:\-–—]?\s*(.*)$",  # noqa: RUF001
    re.IGNORECASE,
)

_WHITESPACE_RE = re.compile(r"\s+")

# Sentinel inserted in place of ``<br>`` before extracting text. The Private
# Use Area codepoint is guaranteed not to appear in real filings and is not
# whitespace, so BeautifulSoup's ``strip=True`` leaves it intact for the
# post-extraction split.
_BR_SENTINEL = ""

# Tags whose text content forms one paragraph. ``body`` is included as a
# fallback so unusual filings with no inner block structure still produce
# some output. ``td``/``th`` are deliberately omitted — emitting one
# paragraph per cell shatters tabular rows; emitting one per ``tr`` keeps
# the row together.
_BLOCK_TAGS = (
    "p",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "tr",
    "blockquote",
    "pre",
    "article",
    "section",
    "header",
    "footer",
    "body",
)

PREAMBLE_LABEL = "preamble"


@dataclass(frozen=True, slots=True)
class Section:
    """A contiguous run of paragraphs sharing a section label.

    ``label`` is a lowercase, slug-like identifier such as ``"item_1a"``,
    ``"item_5.07"``, ``"part_ii"``, or :data:`PREAMBLE_LABEL` for content
    that precedes the first detected heading.
    """

    label: str
    title: str | None
    paragraphs: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Concatenated section body with one blank line between paragraphs."""

        return "\n\n".join(self.paragraphs)

    @property
    def char_count(self) -> int:
        return sum(len(p) for p in self.paragraphs)


def parse_filing_html(html: bytes | str) -> list[str]:
    """Return the filing's visible text as a list of paragraphs.

    Each element of the returned list is one logical paragraph: whitespace
    is collapsed, scripts/styles are dropped, and reading order is preserved.

    The parser walks leaf-most block elements (``<p>``, ``<div>``, ``<h*>``,
    ``<li>``, ``<tr>``, ...). Inline elements (``<span>``, ``<b>``, ``<a>``,
    inline-XBRL) are kept inline within their containing paragraph. ``<br>``
    is promoted to a paragraph break so soft line breaks split correctly.
    """

    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "head"]):
        tag.decompose()

    for br in soup.find_all("br"):
        br.replace_with(_BR_SENTINEL)

    paragraphs: list[str] = []
    for elem in soup.find_all(_BLOCK_TAGS):
        # Only leaf block elements produce paragraphs — parent blocks
        # would otherwise double-count their children's text.
        if elem.find(_BLOCK_TAGS):
            continue
        raw = elem.get_text(" ", strip=True)
        for line in raw.split(_BR_SENTINEL):
            cleaned = _WHITESPACE_RE.sub(" ", line).strip()
            if cleaned:
                paragraphs.append(cleaned)

    return paragraphs


def detect_sections(paragraphs: list[str]) -> list[Section]:
    """Group paragraphs into sections by detecting ``Item``/``Part`` headings.

    Paragraphs that precede the first detected heading land in a single
    :data:`PREAMBLE_LABEL` section. When the same label is detected more
    than once (typically once in the table of contents and once in the
    body), only the longest occurrence is retained.
    """

    sections: list[Section] = []
    current = Section(label=PREAMBLE_LABEL, title=None, paragraphs=[])
    sections.append(current)

    for paragraph in paragraphs:
        heading = _match_heading(paragraph)
        if heading is not None:
            label, title = heading
            current = Section(label=label, title=title, paragraphs=[])
            sections.append(current)
            continue
        current.paragraphs.append(paragraph)

    return _dedupe_sections(sections)


def _match_heading(paragraph: str) -> tuple[str, str | None] | None:
    """Return ``(label, title)`` if ``paragraph`` looks like a section heading."""

    if len(paragraph) > _MAX_HEADING_CHARS:
        return None

    match = _ITEM_RE.match(paragraph)
    if match is not None:
        number = match.group(1).lower()
        title = _clean_title(match.group(2))
        return f"item_{number}", title

    match = _PART_RE.match(paragraph)
    if match is not None:
        roman = match.group(1).lower()
        title = _clean_title(match.group(2))
        return f"part_{roman}", title

    return None


def _clean_title(raw: str) -> str | None:
    cleaned = _WHITESPACE_RE.sub(" ", raw).strip(" .:-–—")  # noqa: RUF001
    return cleaned or None


def _dedupe_sections(sections: list[Section]) -> list[Section]:
    """Keep only the longest occurrence of each label, preserving order."""

    longest: dict[str, Section] = {}
    for section in sections:
        if section.label == PREAMBLE_LABEL and not section.paragraphs:
            # Drop an empty preamble (filings where the very first paragraph
            # is already a section heading).
            continue
        existing = longest.get(section.label)
        if existing is None or section.char_count > existing.char_count:
            longest[section.label] = section

    # Re-order to match the first occurrence in the original list so the
    # output reads top-to-bottom as the filing does.
    seen: set[str] = set()
    ordered: list[Section] = []
    for section in sections:
        if section.label in seen:
            continue
        kept = longest.get(section.label)
        if kept is None:
            continue
        ordered.append(kept)
        seen.add(section.label)
    return ordered


__all__ = [
    "PREAMBLE_LABEL",
    "Section",
    "detect_sections",
    "parse_filing_html",
]
