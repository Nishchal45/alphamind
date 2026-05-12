"""End-to-end chunking pipeline.

Steps:

1. HTML to plain text (:mod:`alphamind.retrieval.chunking.text`).
2. Detect 10-K / 10-Q section boundaries
   (:mod:`alphamind.retrieval.chunking.sections`).
3. Token-aware sliding window inside each section so a single chunk
   never spans two named sections — keeping the section label honest.

Chunks below ``ChunkingConfig.min_tokens`` are dropped to avoid polluting
the index with page numbers, signature blocks, and other one-line junk.
"""

from __future__ import annotations

from alphamind.retrieval.chunking.base import Chunk, ChunkingConfig
from alphamind.retrieval.chunking.sections import detect_sections
from alphamind.retrieval.chunking.splitter import TokenAwareSplitter, count_tokens
from alphamind.retrieval.chunking.text import html_to_text


class ChunkingPipeline:
    """Orchestrates HTML → sections → token-aware splits → :class:`Chunk` list."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self._config = config or ChunkingConfig()
        self._splitter = TokenAwareSplitter(
            target_tokens=self._config.target_tokens,
            overlap_tokens=self._config.overlap_tokens,
        )

    @property
    def config(self) -> ChunkingConfig:
        return self._config

    def chunk_text(self, text: str) -> list[Chunk]:
        """Chunk a pre-extracted plain-text document."""

        if not text:
            return []

        sections = detect_sections(text)
        chunks: list[Chunk] = []
        ordinal = 0

        for section in sections:
            section_text = text[section.start : section.end]
            for offset_start, offset_end, token_count in self._splitter.split(section_text):
                if token_count < self._config.min_tokens:
                    continue
                char_start = section.start + offset_start
                char_end = section.start + offset_end
                chunks.append(
                    Chunk(
                        ordinal=ordinal,
                        text=text[char_start:char_end],
                        token_count=token_count,
                        section=None if section.name == "Preamble" else section.name,
                        char_start=char_start,
                        char_end=char_end,
                    )
                )
                ordinal += 1

        return chunks

    def chunk_html(self, html: str | bytes) -> list[Chunk]:
        """Chunk an HTML body fetched from EDGAR."""

        return self.chunk_text(html_to_text(html))


__all__ = ["Chunk", "ChunkingConfig", "ChunkingPipeline", "count_tokens"]
