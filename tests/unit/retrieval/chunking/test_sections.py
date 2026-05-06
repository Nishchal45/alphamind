"""Tests for 10-K section detection."""

from __future__ import annotations

from alphamind.retrieval.chunking.sections import detect_sections


SAMPLE_10K = """
PART I

Item 1. Business

We design, manufacture, and market computers and other devices.

Item 1A. Risk Factors

The following risk factors could materially affect our business.

Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations

Net sales increased 5% year over year.

Item 8. Financial Statements and Supplementary Data

See accompanying notes.
"""


def test_detects_canonical_items() -> None:
    sections = detect_sections(SAMPLE_10K)

    names = [s.name for s in sections]

    # All four items should appear in document order, possibly with a Preamble first.
    assert "Item 1. Business" in names
    assert "Item 1A. Risk Factors" in names
    assert "Item 7. Management's Discussion and Analysis" in names
    assert "Item 8. Financial Statements" in names

    # Document-order monotonicity: starts are non-decreasing.
    starts = [s.start for s in sections]
    assert starts == sorted(starts)


def test_sections_partition_the_document() -> None:
    sections = detect_sections(SAMPLE_10K)

    # First section starts at 0; last ends at len(text); adjacent ranges meet.
    assert sections[0].start == 0
    assert sections[-1].end == len(SAMPLE_10K)
    for prev, curr in zip(sections, sections[1:], strict=False):
        assert prev.end == curr.start


def test_documents_without_items_get_single_preamble() -> None:
    text = "Some letter to shareholders that doesn't follow Item conventions."

    sections = detect_sections(text)

    assert len(sections) == 1
    assert sections[0].name == "Preamble"
    assert sections[0].start == 0
    assert sections[0].end == len(text)


def test_empty_text_returns_no_sections() -> None:
    assert detect_sections("") == []


def test_table_of_contents_does_not_create_duplicate_sections() -> None:
    """A 10-K with a TOC will mention each Item heading twice. Detector
    keeps the first occurrence — the TOC entry — which still partitions
    the document correctly because following sections start at later
    offsets."""

    text = """Table of Contents
    Item 1. Business ............ 4
    Item 1A. Risk Factors ....... 12
    Item 7. Management's Discussion and Analysis of Financial Condition

Item 1. Business

This is the actual business section.

Item 1A. Risk Factors

These are the actual risk factors.

Item 7. Management's Discussion and Analysis

Here is the actual MD&A.
"""

    sections = detect_sections(text)
    names = [s.name for s in sections]

    # Each canonical name appears exactly once.
    assert names.count("Item 1. Business") == 1
    assert names.count("Item 1A. Risk Factors") == 1
    assert names.count("Item 7. Management's Discussion and Analysis") == 1
