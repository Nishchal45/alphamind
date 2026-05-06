"""Shared types for the chunking pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Knobs that control the chunker.

    Defaults are tuned for SEC filings:

    - ``target_tokens=512`` matches typical embedding-model context windows
      (bge-small, e5-base, MiniLM all comfortably handle 512 tokens).
    - ``overlap_ratio=0.15`` preserves cross-chunk context without exploding
      storage cost. 15% means a 512-token chunk overlaps the next by ~77 tokens.
    - ``min_tokens=64`` drops trivially short chunks (signature lines, page
      numbers) that pollute the index without adding signal.
    """

    target_tokens: int = 512
    overlap_ratio: float = 0.15
    min_tokens: int = 64

    def __post_init__(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if not 0 <= self.overlap_ratio < 1:
            raise ValueError("overlap_ratio must be in [0, 1)")
        if self.min_tokens < 0:
            raise ValueError("min_tokens must be non-negative")
        if self.min_tokens > self.target_tokens:
            raise ValueError("min_tokens cannot exceed target_tokens")

    @property
    def overlap_tokens(self) -> int:
        return int(self.target_tokens * self.overlap_ratio)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single chunk produced by the pipeline.

    ``ordinal`` is a 0-based position within the filing, used as the stable
    sort key for retrieval display and as the natural key for upserts.
    """

    ordinal: int
    text: str
    token_count: int
    section: str | None
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
