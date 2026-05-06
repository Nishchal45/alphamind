"""Factory that returns the configured storage backend as a singleton."""

from __future__ import annotations

from functools import lru_cache

from alphamind.config import get_settings
from alphamind.storage.base import StorageBackend, StorageError
from alphamind.storage.local import LocalFilesystemStorage


@lru_cache(maxsize=1)
def get_storage() -> StorageBackend:
    """Return the process-wide storage backend instance."""

    settings = get_settings()
    backend = settings.storage_backend.lower()

    if backend == "local":
        return LocalFilesystemStorage(root=settings.storage_local_path)

    raise StorageError(f"unsupported storage backend: {backend!r}")


__all__ = ["get_storage"]
