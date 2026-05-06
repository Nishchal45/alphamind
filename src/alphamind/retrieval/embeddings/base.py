"""Embedder protocol shared by every concrete implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


class EmbedderError(RuntimeError):
    """Raised when the embedder cannot encode an input."""


@runtime_checkable
class Embedder(Protocol):
    """Encodes one or more passages into fixed-length unit-norm vectors.

    Implementations must be safe to call concurrently and must return
    vectors that match :attr:`dim`. Vectors are normalised to unit length
    so cosine similarity reduces to a dot product, which the search
    pipeline relies on.
    """

    @property
    def dim(self) -> int:
        """Output dimension. Must match :data:`alphamind.models.EMBEDDING_DIM`."""
        ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a batch of strings, preserving input order."""
        ...
