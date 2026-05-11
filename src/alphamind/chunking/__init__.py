"""Filing-body chunking for downstream embedding and retrieval.

The public surface is:

- :func:`parse_filing_html` — HTML/XBRL bytes to a flat list of text paragraphs.
- :func:`detect_sections` — group paragraphs into 10-K/10-Q/8-K sections.
- :func:`chunk_text` — generic paragraph-aware text splitter with overlap.
- :func:`chunk_filing` — end-to-end: HTML in, :class:`Chunk` list out.

The chunker is deliberately model-agnostic. It targets a character budget
rather than a token budget so it does not couple to a specific tokenizer;
the budget is configurable and sized conservatively for common embedding
models (~1000 tokens at ~4 chars/token for English prose).
"""

from __future__ import annotations

from alphamind.chunking.chunker import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    Chunk,
    chunk_filing,
    chunk_text,
)
from alphamind.chunking.parser import (
    Section,
    detect_sections,
    parse_filing_html,
)
from alphamind.chunking.service import (
    ChunkBatchResult,
    ChunkIngestResult,
    chunk_bodies_for_cik,
    chunk_filing_document,
)

__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_OVERLAP_CHARS",
    "Chunk",
    "ChunkBatchResult",
    "ChunkIngestResult",
    "Section",
    "chunk_bodies_for_cik",
    "chunk_filing",
    "chunk_filing_document",
    "chunk_text",
    "detect_sections",
    "parse_filing_html",
]
