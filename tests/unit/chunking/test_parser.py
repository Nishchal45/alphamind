"""Unit tests for the filing HTML parser."""

from __future__ import annotations

from alphamind.chunking.parser import (
    PREAMBLE_LABEL,
    detect_sections,
    parse_filing_html,
)


def test_parse_filing_html_extracts_paragraphs_in_reading_order() -> None:
    html = b"""
    <html><body>
        <p>First paragraph.</p>
        <p>Second paragraph.</p>
        <div>Third block.</div>
    </body></html>
    """

    paragraphs = parse_filing_html(html)

    assert paragraphs == ["First paragraph.", "Second paragraph.", "Third block."]


def test_parse_filing_html_drops_scripts_styles_and_noscript() -> None:
    html = b"""
    <html>
      <head><style>body{color:red}</style></head>
      <body>
        <script>alert('x')</script>
        <noscript>JS required</noscript>
        <p>Visible body.</p>
      </body>
    </html>
    """

    paragraphs = parse_filing_html(html)

    assert paragraphs == ["Visible body."]


def test_parse_filing_html_collapses_whitespace() -> None:
    """Whitespace inside a single block (including source newlines) is collapsed."""

    html = b"<p>Spaced\n\n\tout   text\nhere.</p>"

    paragraphs = parse_filing_html(html)

    # HTML treats source newlines and tabs as ordinary whitespace, so the
    # whole ``<p>`` is one paragraph with internal whitespace collapsed.
    assert paragraphs == ["Spaced out text here."]


def test_parse_filing_html_promotes_br_to_paragraph_break() -> None:
    html = b"<p>Line one.<br>Line two.<br/>Line three.</p>"

    paragraphs = parse_filing_html(html)

    assert paragraphs == ["Line one.", "Line two.", "Line three."]


def test_parse_filing_html_handles_inline_xbrl() -> None:
    """Inline-XBRL ``<ix:...>`` tags are passed through as ordinary text."""

    html = b"""
    <html><body>
      <p>Revenue was <ix:nonNumeric name="us-gaap:Revenue">$1.0 billion</ix:nonNumeric>.</p>
    </body></html>
    """

    paragraphs = parse_filing_html(html)

    # lxml normalizes the inline-XBRL element but its text content survives.
    assert paragraphs == ["Revenue was $1.0 billion ."]


def test_detect_sections_groups_paragraphs_by_item_heading() -> None:
    paragraphs = [
        "Item 1. Business",
        "We make computers.",
        "Our segments are diverse.",
        "Item 1A. Risk Factors",
        "Risk one.",
        "Risk two.",
    ]

    sections = detect_sections(paragraphs)

    labels = [s.label for s in sections]
    assert labels == ["item_1", "item_1a"]

    business = sections[0]
    assert business.title == "Business"
    assert business.paragraphs == ["We make computers.", "Our segments are diverse."]

    risk = sections[1]
    assert risk.title == "Risk Factors"
    assert risk.paragraphs == ["Risk one.", "Risk two."]


def test_detect_sections_preserves_preamble() -> None:
    paragraphs = [
        "Some boilerplate before the first item.",
        "More cover-page text.",
        "Item 1. Business",
        "Body.",
    ]

    sections = detect_sections(paragraphs)

    assert sections[0].label == PREAMBLE_LABEL
    assert sections[0].paragraphs == [
        "Some boilerplate before the first item.",
        "More cover-page text.",
    ]
    assert sections[1].label == "item_1"


def test_detect_sections_drops_empty_preamble() -> None:
    paragraphs = ["Item 1. Business", "Body content."]

    sections = detect_sections(paragraphs)

    assert [s.label for s in sections] == ["item_1"]


def test_detect_sections_dedupes_toc_against_body() -> None:
    """When ``Item N`` appears twice, the longer occurrence wins.

    Real filings repeat every item label in their table of contents at the
    top of the document, then again where the actual section starts. We
    keep the body (large), not the TOC (small).
    """

    paragraphs = [
        "Table of Contents",
        "Item 1. Business",
        "Item 1A. Risk Factors",
        # The body of Item 1 follows with substantive text.
        "Item 1. Business",
        "We design, manufacture and market consumer electronics.",
        "Our products include phones, tablets, and personal computers.",
        "Our research and development efforts focus on next-generation devices.",
        "Item 1A. Risk Factors",
        "Our business is subject to a variety of risks including supply-chain disruption.",
        "We face significant competition in every market we operate in.",
    ]

    sections = detect_sections(paragraphs)

    labels = [s.label for s in sections]
    assert labels == ["preamble", "item_1", "item_1a"]

    item_1 = next(s for s in sections if s.label == "item_1")
    assert any("consumer electronics" in p for p in item_1.paragraphs)
    # The TOC version, which had zero body paragraphs, is gone.
    assert len(item_1.paragraphs) == 3


def test_detect_sections_recognises_part_and_dotted_items() -> None:
    paragraphs = [
        "Part II. Other Information",
        "Item 5.07 Submission of Matters to a Vote of Security Holders",
        "At the annual meeting, the following matters were submitted.",
        "Item 9.01 Financial Statements and Exhibits",
        "Exhibit 99.1 is attached hereto.",
    ]

    sections = detect_sections(paragraphs)

    labels = [s.label for s in sections]
    assert labels == ["part_ii", "item_5.07", "item_9.01"]


def test_detect_sections_ignores_item_mentions_in_body_prose() -> None:
    """A long sentence that happens to start with 'Item 1' is not a heading."""

    paragraphs = [
        "Item 1. Business",
        # 250+ chars of prose; mentions 'Item' but is plainly not a heading.
        "Item 1 of this report is intended to summarize our business and its "
        "principal segments. We organize our internal reporting around three "
        "operating units, each of which manages its own product roadmap and "
        "go-to-market strategy. The discussion that follows draws on management's "
        "current expectations and is subject to risks described elsewhere.",
    ]

    sections = detect_sections(paragraphs)

    assert [s.label for s in sections] == ["item_1"]
    assert len(sections[0].paragraphs) == 1
