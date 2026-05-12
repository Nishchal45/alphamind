"""Tests for HTML to plain-text rendering."""

from __future__ import annotations

from alphamind.retrieval.chunking.text import html_to_text


def test_strips_script_and_style() -> None:
    html = """
    <html>
      <head><style>p { color: red }</style></head>
      <body>
        <script>alert('boo')</script>
        <p>Real content.</p>
      </body>
    </html>
    """

    out = html_to_text(html)

    assert "alert" not in out
    assert "color" not in out
    assert "Real content." in out


def test_preserves_paragraph_breaks() -> None:
    html = "<p>First paragraph.</p><p>Second paragraph.</p>"

    out = html_to_text(html)

    # Two paragraphs should remain on separate lines.
    assert "First paragraph." in out
    assert "Second paragraph." in out
    lines = [line for line in out.split("\n") if line.strip()]
    assert lines == ["First paragraph.", "Second paragraph."]


def test_collapses_whitespace_and_blank_lines() -> None:
    html = "<p>Tabs\tand    spaces.</p>\n\n\n\n<p>Next.</p>"

    out = html_to_text(html)

    assert "Tabs and spaces." in out
    assert "\n\n\n" not in out


def test_normalises_unicode() -> None:
    # The literal characters are U+00A0 (NBSP), U+201C/201D (smart
    # quotes); written via \u escapes so the source stays plain ASCII.
    html = "<p>Apple\u00a0Inc. \u201cdoing well\u201d</p>"

    out = html_to_text(html)

    # NFKC turns the non-breaking space into a regular space.
    assert "Apple Inc." in out
    assert "doing well" in out


def test_empty_input_yields_empty_string() -> None:
    assert html_to_text("") == ""
    assert html_to_text("<html></html>") == ""
