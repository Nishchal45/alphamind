"""HTML to plain text conversion for SEC filings.

EDGAR primary documents are mostly HTML with inline XBRL tags. We strip
the markup but keep paragraph and list structure so the section detector
downstream has linebreaks to anchor on.
"""

from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup
from bs4.element import Tag

# Tags whose text content is irrelevant for retrieval (scripts, styles,
# inline metadata). Stripped before serialisation.
_STRIPPED_TAGS = frozenset({"script", "style", "noscript", "head", "meta", "link"})

# Block-level tags that imply a paragraph break in the rendered document.
# We turn each into a single newline so the splitter has natural boundaries.
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "tr",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "section",
        "article",
        "header",
        "footer",
        "table",
        "blockquote",
    }
)

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def html_to_text(html: str | bytes) -> str:
    """Render ``html`` to plain text, preserving paragraph structure.

    Multiple consecutive blank lines collapse to a single blank line. Inline
    whitespace collapses to single spaces. Unicode is NFKC-normalised so
    smart quotes, non-breaking spaces, and ligatures behave predictably for
    downstream tokenisation.
    """

    soup = BeautifulSoup(html, "lxml")

    for tag_name in _STRIPPED_TAGS:
        for node in soup.find_all(tag_name):
            node.decompose()

    # Append a sentinel newline to every block-level tag so get_text()
    # produces line-broken output. ``Tag.append`` is the documented way.
    for node in soup.find_all(True):
        if isinstance(node, Tag) and node.name in _BLOCK_TAGS:
            node.append("\n")

    raw = soup.get_text(separator=" ")
    normalised = unicodedata.normalize("NFKC", raw)

    # Collapse runs of inline whitespace, then runs of blank lines.
    lines = (
        _WHITESPACE_RE.sub(" ", line).strip() for line in normalised.split("\n")
    )
    text = "\n".join(line for line in lines if line)
    return _BLANK_LINES_RE.sub("\n\n", text).strip()
