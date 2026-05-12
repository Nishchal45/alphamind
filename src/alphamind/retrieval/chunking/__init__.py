"""Chunking pipeline for filing bodies.

The pipeline is intentionally three modules so each piece is replaceable:

- :mod:`text` — HTML to plain text, preserving paragraph structure.
- :mod:`sections` — detects 10-K / 10-Q / 8-K section boundaries (Item 1A
  Risk Factors, Item 7 MD&A, etc.) so chunks don't straddle them.
- :mod:`splitter` — token-aware sliding window with overlap.

:class:`alphamind.retrieval.chunking.pipeline.ChunkingPipeline` ties them
together. Applications import from there; the lower-level modules are kept
public so they can be unit-tested independently.
"""

from __future__ import annotations

from alphamind.retrieval.chunking.base import Chunk, ChunkingConfig
from alphamind.retrieval.chunking.pipeline import ChunkingPipeline
from alphamind.retrieval.chunking.sections import Section, detect_sections
from alphamind.retrieval.chunking.splitter import TokenAwareSplitter, count_tokens
from alphamind.retrieval.chunking.text import html_to_text

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "ChunkingPipeline",
    "Section",
    "TokenAwareSplitter",
    "count_tokens",
    "detect_sections",
    "html_to_text",
]
