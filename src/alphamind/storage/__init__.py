"""Object-storage abstraction for filing bodies and other large blobs.

Filings range from a few kilobytes (an 8-K) to tens of megabytes (a 10-K with
exhibits). Inlining bodies in Postgres bloats page caches and slows migrations,
so they live behind this storage abstraction instead. ADR 0004 covers the
design.

The :class:`StorageBackend` protocol is intentionally narrow — ``put``,
``get``, ``exists`` — so swapping the local-filesystem default for an
S3-compatible backend later is a config change rather than a code change.
"""

from __future__ import annotations

from alphamind.storage.base import StorageBackend, StorageError
from alphamind.storage.factory import get_storage
from alphamind.storage.local import LocalFilesystemStorage

__all__ = [
    "LocalFilesystemStorage",
    "StorageBackend",
    "StorageError",
    "get_storage",
]
