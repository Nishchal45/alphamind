"""Local-filesystem storage backend.

Used as the default in development and tests. Production deployments are
expected to swap in an S3-compatible backend without touching call sites
(see :mod:`alphamind.storage.factory`).

Layout under ``root``::

    <root>/<key[:2]>/<key>

Sharding by the first two characters keeps any one directory from accumulating
millions of entries when the corpus grows large. Keys produced by the
ingestion service are SHA-256 hex digests, so the sharding distribution is
uniform.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlparse

from alphamind.storage.base import StorageError

_SHARD_PREFIX_LEN = 2


class LocalFilesystemStorage:
    """Persist blobs under a directory tree on the local filesystem.

    Parameters
    ----------
    root:
        Directory the backend owns. Created if it does not exist.
    """

    SCHEME = "file"

    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        if not key or "/" in key or ".." in key:
            raise StorageError(f"invalid storage key: {key!r}")
        prefix = key[:_SHARD_PREFIX_LEN] if len(key) >= _SHARD_PREFIX_LEN else "_"
        return self._root / prefix / key

    def _uri_for(self, path: Path) -> str:
        return f"{self.SCHEME}://{path}"

    def _path_from_uri(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != self.SCHEME:
            raise StorageError(f"unsupported uri scheme for local backend: {uri!r}")
        # ``urlparse('file:///abs/path').path`` -> ``'/abs/path'``;
        # ``urlparse('file://relpath').path`` -> ``''`` with ``netloc='relpath'``.
        # We always emit absolute paths, so prefer ``path``.
        path_str = parsed.path or parsed.netloc
        return Path(path_str)

    async def put(self, *, key: str, data: bytes) -> str:
        path = self._path_for(key)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic-ish write: stage to a sibling file, then rename. Avoids
            # readers seeing a half-written blob if the process is killed mid-write.
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)

        await asyncio.to_thread(_write)
        return self._uri_for(path)

    async def get(self, uri: str) -> bytes:
        path = self._path_from_uri(uri)
        if not path.exists():
            raise StorageError(f"object not found: {uri!r}")
        return await asyncio.to_thread(path.read_bytes)

    async def exists(self, uri: str) -> bool:
        path = self._path_from_uri(uri)
        return await asyncio.to_thread(path.exists)
