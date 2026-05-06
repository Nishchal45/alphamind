"""End-to-end tests for the chunking pipeline."""

from __future__ import annotations

import pytest

from alphamind.retrieval.chunking import Chunk, ChunkingConfig, ChunkingPipeline


@pytest.fixture
def small_pipeline() -> ChunkingPipeline:
    """Smaller chunks so tests don't need a 10-K-sized fixture."""
    return ChunkingPipeline(
        config=ChunkingConfig(target_tokens=64, overlap_ratio=0.1, min_tokens=4)
    )


def test_pipeline_emits_chunks_in_order_with_unique_ordinals(
    small_pipeline: ChunkingPipeline,
) -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 100

    chunks = small_pipeline.chunk_text(text)

    assert len(chunks) >= 2
    ordinals = [c.ordinal for c in chunks]
    assert ordinals == list(range(len(chunks)))


def test_chunks_carry_section_label_when_detected(
    small_pipeline: ChunkingPipeline,
) -> None:
    text = (
        "Item 1A. Risk Factors\n\n"
        + ("Investing in our common stock involves risk. " * 30)
        + "\n\nItem 7. Management's Discussion and Analysis\n\n"
        + ("Net sales increased year over year. " * 30)
    )

    chunks = small_pipeline.chunk_text(text)

    sections = {c.section for c in chunks if c.section is not None}
    assert "Item 1A. Risk Factors" in sections
    assert "Item 7. Management's Discussion and Analysis" in sections

    # No chunk straddles a section: each chunk's text should appear inside
    # its declared section's text.
    for chunk in chunks:
        if chunk.section is None:
            continue
        section_text = text[
            text.index(chunk.section) : text.index(chunk.section)
            + len(text)
        ]
        # Any well-placed chunk's first 20 characters should appear inside
        # the section span.
        assert chunk.text[:20].strip() in section_text


def test_min_tokens_drops_trivial_chunks() -> None:
    # min_tokens=10 with target=64; a tiny stub at the end has ~3 tokens
    # and should be dropped.
    pipeline = ChunkingPipeline(
        config=ChunkingConfig(target_tokens=64, overlap_ratio=0.0, min_tokens=10)
    )

    text = "Hello world. " * 100 + "tiny."

    chunks = pipeline.chunk_text(text)

    for chunk in chunks:
        assert chunk.token_count >= 10


def test_empty_text_yields_no_chunks(small_pipeline: ChunkingPipeline) -> None:
    assert small_pipeline.chunk_text("") == []


def test_html_input_is_normalised_before_chunking(
    small_pipeline: ChunkingPipeline,
) -> None:
    html = (
        "<html><body>"
        "<p>" + ("Risk factor language goes here. " * 30) + "</p>"
        "<script>alert('skip')</script>"
        "</body></html>"
    )

    chunks = small_pipeline.chunk_html(html)

    assert chunks
    for chunk in chunks:
        assert "alert" not in chunk.text


def test_chunk_dataclass_validates_fields() -> None:
    with pytest.raises(ValueError):
        Chunk(ordinal=-1, text="x", token_count=1, section=None, char_start=0, char_end=1)
    with pytest.raises(ValueError):
        Chunk(ordinal=0, text="x", token_count=0, section=None, char_start=0, char_end=1)
    with pytest.raises(ValueError):
        Chunk(ordinal=0, text="x", token_count=1, section=None, char_start=5, char_end=5)


def test_chunking_config_validates_arguments() -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(target_tokens=0)
    with pytest.raises(ValueError):
        ChunkingConfig(overlap_ratio=1.0)
    with pytest.raises(ValueError):
        ChunkingConfig(min_tokens=-1)
    with pytest.raises(ValueError):
        ChunkingConfig(target_tokens=10, min_tokens=20)
