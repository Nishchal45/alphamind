"""Storage backend protocol shared by every concrete implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class StorageError(RuntimeError):
    """Raised when the underlying storage layer cannot satisfy a request."""


@runtime_checkable
class StorageBackend(Protocol):
    """A minimal byte-blob store.

    Implementations are expected to be safe to call concurrently from multiple
    asyncio tasks. ``put`` is idempotent: writing the same ``key`` twice with
    the same bytes is a no-op from the caller's perspective.
    """

    async def put(self, *, key: str, data: bytes) -> str:
        """Persist ``data`` under ``key`` and return an opaque URI to read it back."""
        ...

    async def get(self, uri: str) -> bytes:
        """Fetch the bytes previously stored at ``uri``.

        Raises :class:`StorageError` if the URI is not known to this backend.
        """
        ...

    async def exists(self, uri: str) -> bool:
        """Return ``True`` if ``uri`` resolves to readable bytes."""
        ...
