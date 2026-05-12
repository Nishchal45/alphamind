"""Detect 10-K / 10-Q / 8-K section boundaries in plain-text filings.

Sections matter because risk factors (Item 1A) and MD&A (Item 7) are
where signal lives. A retrieval call that asks "what does AAPL say about
China revenue concentration" should ideally land on the MD&A section,
not on a boilerplate forward-looking-statements disclaimer. Tagging
chunks with their section name lets the search layer filter or boost
on it cheaply.

The detector is regex-based — fast, no model dependency, recall is high
on filings that follow standard formatting. Filings that don't (rare,
older, pre-2003 ish) get a single ``None``-section span. The downstream
chunker still works in that case; section metadata just stays ``None``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Section:
    """Half-open ``[start, end)`` byte range of a single named section."""

    name: str
    start: int
    end: int


# 10-K / 10-Q canonical Item headings. Order matters only for display;
# detection sorts by position in the document.
_ITEM_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"item\s+1a\.?\s+risk\s+factors", "Item 1A. Risk Factors"),
    (r"item\s+1b\.?\s+unresolved\s+staff\s+comments", "Item 1B. Unresolved Staff Comments"),
    (r"item\s+1\.?\s+business", "Item 1. Business"),
    (r"item\s+2\.?\s+properties", "Item 2. Properties"),
    (r"item\s+3\.?\s+legal\s+proceedings", "Item 3. Legal Proceedings"),
    (r"item\s+4\.?\s+mine\s+safety", "Item 4. Mine Safety Disclosures"),
    (
        r"item\s+5\.?\s+market\s+for\s+registrant",
        "Item 5. Market for Registrant's Common Equity",
    ),
    (r"item\s+6\.?\s+(\[reserved\]|selected\s+financial)", "Item 6. Selected Financial Data"),
    (
        r"item\s+7a\.?\s+quantitative\s+and\s+qualitative",
        "Item 7A. Quantitative and Qualitative Disclosures",
    ),
    (
        # The curly apostrophe U+2019 is what almost every real EDGAR
        # filing uses; the plain ASCII apostrophe is the rarer fallback.
        "item\\s+7\\.?\\s+management(?:'|\u2019)s\\s+discussion",
        "Item 7. Management's Discussion and Analysis",
    ),
    (r"item\s+8\.?\s+financial\s+statements", "Item 8. Financial Statements"),
    (
        r"item\s+9a\.?\s+controls\s+and\s+procedures",
        "Item 9A. Controls and Procedures",
    ),
    (
        r"item\s+9\.?\s+changes\s+in\s+and\s+disagreements",
        "Item 9. Changes in and Disagreements with Accountants",
    ),
    (
        r"item\s+10\.?\s+directors,?\s+executive\s+officers",
        "Item 10. Directors, Executive Officers and Corporate Governance",
    ),
    (r"item\s+11\.?\s+executive\s+compensation", "Item 11. Executive Compensation"),
    (
        r"item\s+12\.?\s+security\s+ownership",
        "Item 12. Security Ownership of Certain Beneficial Owners",
    ),
    (
        r"item\s+13\.?\s+certain\s+relationships",
        "Item 13. Certain Relationships and Related Transactions",
    ),
    (r"item\s+14\.?\s+principal\s+account", "Item 14. Principal Accountant Fees and Services"),
    (r"item\s+15\.?\s+exhibits?", "Item 15. Exhibits"),
)

_COMPILED_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rf"(?im)^\s*{pattern}\b.*$", re.MULTILINE), name)
    for pattern, name in _ITEM_PATTERNS
)


def detect_sections(text: str) -> list[Section]:
    """Return contiguous ``Section`` ranges covering ``text``.

    The list is sorted by ``start`` and partitions the document: the first
    section begins at byte 0 (``"Preamble"`` if no Item header precedes
    real content) and the last section ends at ``len(text)``.
    """

    if not text:
        return []

    raw_hits: list[tuple[int, str]] = []
    for pattern, name in _COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            raw_hits.append((match.start(), name))

    if not raw_hits:
        return [Section(name="Preamble", start=0, end=len(text))]

    raw_hits.sort()
    # Multiple matches for the same name are common (TOC + the section
    # itself). Keep the first occurrence of each name in document order.
    seen: set[str] = set()
    headers: list[tuple[int, str]] = []
    for start, name in raw_hits:
        if name in seen:
            continue
        seen.add(name)
        headers.append((start, name))

    headers.sort()

    sections: list[Section] = []
    if headers[0][0] > 0:
        sections.append(Section(name="Preamble", start=0, end=headers[0][0]))

    for i, (start, name) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        sections.append(Section(name=name, start=start, end=end))

    return sections
