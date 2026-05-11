"""Deterministic hash-based embedder for tests and local development.

This is a *real* embedder in the sense that it implements the protocol and
produces stable, L2-normalised vectors. It is *not* a semantically useful
embedder — there is no model and no training signal. Use it to exercise
the embedding/retrieval plumbing while a real backend is wired up.

The vector for a given text is built bag-of-words style: each whitespace-
separated token is mapped to a fixed index via a SHA-256 hash and adds a
sign-flipped unit to the corresponding coordinate. The sum is then
L2-normalised so cosine similarity is well defined. Two texts that share
many tokens get visibly similar vectors; unrelated texts get nearly
orthogonal vectors — enough structure to test retrieval mechanics.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re

from alphamind.embeddings.base import Vector

DEFAULT_DIMENSION = 768

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class DeterministicEmbedder:
    """Hash-based embedder that produces stable vectors without a model.

    Parameters
    ----------
    dimension:
        Length of each returned vector. Defaults to 384 to match
        ``sentence-transformers/all-MiniLM-L6-v2``, the most likely first
        real backend.
    """

    def __init__(self, *, dimension: int = DEFAULT_DIMENSION) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension
        self._model_name = f"deterministic-hash-{dimension}"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[Vector]:
        # Pure CPU work; hop to a worker thread so we don't block the event
        # loop on a large batch.
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[Vector]:
        return [self._encode(text) for text in texts]

    def _encode(self, text: str) -> Vector:
        vector = [0.0] * self._dimension
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            # Use the next byte to flip sign so colliding tokens don't always
            # reinforce; weak but cheap signal differentiation.
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]


__all__ = ["DEFAULT_DIMENSION", "DeterministicEmbedder"]
