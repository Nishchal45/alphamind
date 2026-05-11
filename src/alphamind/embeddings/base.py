"""Embedder protocol shared by every concrete implementation.

Backends differ along three dimensions: vector dimension, model identity
(used for staleness checks), and physical implementation (local model,
hosted API, ...). The protocol fixes the interface so call sites — the
embedding service and any future retrieval code — can swap backends via
:mod:`alphamind.embeddings.factory` without touching their own code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

Vector = list[float]


class EmbedderError(RuntimeError):
    """Raised when an embedder cannot satisfy a request."""


@runtime_checkable
class Embedder(Protocol):
    """A batch text-to-vector encoder.

    Implementations must be safe to call concurrently from multiple asyncio
    tasks. The vectors returned for two equal input strings must compare
    equal — embedding determinism lets the service skip work and lets tests
    assert against expected values.
    """

    @property
    def model_name(self) -> str:
        """Stable identifier for this embedder, e.g. ``"fake-hash-384"``.

        Persisted alongside each embedding so the service can detect when
        the configured backend has changed and trigger a re-embed.
        """
        ...

    @property
    def dimension(self) -> int:
        """Number of components in each returned :data:`Vector`."""
        ...

    async def embed(self, texts: list[str]) -> list[Vector]:
        """Encode ``texts`` into vectors. Order is preserved.

        Raises :class:`EmbedderError` if the backend cannot encode a batch.
        """
        ...


__all__ = ["Embedder", "EmbedderError", "Vector"]
