"""Unit tests for the paragraph-aware chunker."""

from __future__ import annotations

from itertools import pairwise

import pytest

from alphamind.chunking.chunker import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    Chunk,
    chunk_filing,
    chunk_text,
)


def test_chunk_text_returns_single_chunk_when_under_budget() -> None:
    text = "Paragraph one.\n\nParagraph two."

    chunks = chunk_text(text, max_chars=1000, overlap_chars=100)

    assert chunks == ["Paragraph one.\n\nParagraph two."]


def test_chunk_text_returns_empty_list_for_empty_input() -> None:
    assert chunk_text("", max_chars=1000, overlap_chars=100) == []
    assert chunk_text("   \n\n   ", max_chars=1000, overlap_chars=100) == []


def test_chunk_text_respects_max_chars() -> None:
    paragraphs = [f"Paragraph {i} with some filler text to add length." for i in range(20)]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text, max_chars=200, overlap_chars=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 200


def test_chunk_text_includes_overlap_between_consecutive_chunks() -> None:
    paragraphs = [f"Paragraph number {i}." for i in range(30)]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text, max_chars=120, overlap_chars=40)

    assert len(chunks) > 1
    for prev, curr in pairwise(chunks):
        # The overlap is paragraph-aligned, so at least one short paragraph
        # from the end of the previous chunk should reappear in the next.
        assert any(p in curr[:80] for p in prev.split("\n\n")[-2:]), (
            f"no overlap between\n{prev!r}\nand\n{curr!r}"
        )


def test_chunk_text_splits_oversize_paragraph_on_sentence_boundaries() -> None:
    sentence = "This is a single sentence with enough words to matter. "
    long_paragraph = sentence * 20  # ~1100 chars, no internal paragraph break

    chunks = chunk_text(long_paragraph, max_chars=200, overlap_chars=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 200
    # Sentence boundaries are preserved: every chunk should end with
    # terminal punctuation (allowing trailing whitespace).
    for chunk in chunks:
        assert chunk.rstrip().endswith((".", "!", "?"))


def test_chunk_text_falls_back_to_word_split_for_giant_sentence() -> None:
    giant_sentence = "alpha bravo charlie delta echo " * 100  # ~3000 chars, one "sentence"

    chunks = chunk_text(giant_sentence, max_chars=150, overlap_chars=30)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 150


def test_chunk_text_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        chunk_text("hello", max_chars=0, overlap_chars=0)
    with pytest.raises(ValueError, match="overlap_chars"):
        chunk_text("hello", max_chars=100, overlap_chars=-1)
    with pytest.raises(ValueError, match="overlap_chars"):
        chunk_text("hello", max_chars=100, overlap_chars=100)


def test_chunk_text_defaults_round_trip_small_text() -> None:
    text = "Just a sentence."
    chunks = chunk_text(text, max_chars=DEFAULT_MAX_CHARS, overlap_chars=DEFAULT_OVERLAP_CHARS)
    assert chunks == [text]


def test_chunk_filing_emits_chunks_with_section_metadata() -> None:
    html = b"""
    <html><body>
      <p>Item 1. Business</p>
      <p>We design and sell widgets to enterprises worldwide.</p>
      <p>Our segments include software, services, and hardware.</p>
      <p>Item 1A. Risk Factors</p>
      <p>Customer concentration is a material risk.</p>
      <p>Macroeconomic conditions may affect demand.</p>
    </body></html>
    """

    chunks = chunk_filing(html, max_chars=500, overlap_chars=50)

    assert len(chunks) >= 2
    assert chunks[0].section_label == "item_1"
    assert chunks[0].section_title == "Business"
    assert chunks[0].chunk_index == 0
    assert "widgets" in chunks[0].text

    risk_chunks = [c for c in chunks if c.section_label == "item_1a"]
    assert risk_chunks
    assert risk_chunks[0].section_title == "Risk Factors"


def test_chunk_filing_indexes_chunks_globally_in_reading_order() -> None:
    paragraphs = [f"Filler paragraph number {i}." for i in range(50)]
    body = "\n\n".join(paragraphs)
    html = f"""
    <html><body>
      <p>Item 1. Business</p>
      <p>{body}</p>
      <p>Item 1A. Risk Factors</p>
      <p>Single short risk paragraph.</p>
    </body></html>
    """.encode()

    chunks = chunk_filing(html, max_chars=300, overlap_chars=50)

    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_filing_chunk_count_matches_char_count_field() -> None:
    html = b"<html><body><p>Item 1. Business</p><p>Short body paragraph.</p></body></html>"

    chunks = chunk_filing(html)

    assert chunks
    for chunk in chunks:
        assert chunk.char_count == len(chunk.text)


def test_chunk_is_immutable() -> None:
    chunk = Chunk(
        section_label="item_1",
        section_title="Business",
        chunk_index=0,
        text="hello",
        char_count=5,
    )
    with pytest.raises((AttributeError, TypeError)):
        chunk.chunk_index = 99  # type: ignore[misc]


def test_chunk_filing_returns_empty_for_documents_with_no_text() -> None:
    html = b"<html><body><style>p{color:red}</style></body></html>"

    assert chunk_filing(html) == []
