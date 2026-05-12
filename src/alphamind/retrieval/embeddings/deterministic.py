"""Deterministic, seeded embedder for tests and dev.

Real embeddings come from a sentence-transformer (Phase 3 work). For
development and tests we want something that:

1. Is deterministic — the same string always maps to the same vector.
2. Has zero external dependencies — no model download, no GPU, no network.
3. Doesn't claim to be semantically meaningful — it isn't, and the
   class name says so.

This implementation seeds a NumPy RNG with the SHA-256 of the input,
samples a Gaussian vector, and L2-normalises. Two identical strings get
identical vectors; two different strings get pseudo-random orthogonal
vectors. That's enough to exercise the pgvector index, the cosine-distance
query, and the search pipeline end-to-end.

Do not deploy this. Real semantic search needs a real model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from alphamind.models.filing_chunk import EMBEDDING_DIM


class DeterministicHashEmbedder:
    """Hash-seeded RNG embedder. Useful for tests; useless for real retrieval."""

    def __init__(self, *, dim: int = EMBEDDING_DIM) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vector_for(self, text: str) -> list[float]:
        seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()[:8]
        seed = int.from_bytes(seed_bytes, "big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        return [float(x) for x in vec.tolist()]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector_for(text) for text in texts]
