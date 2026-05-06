"""Unit tests for the local-filesystem storage backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphamind.storage.base import StorageError
from alphamind.storage.local import LocalFilesystemStorage

pytestmark = pytest.mark.asyncio


async def test_put_returns_uri_and_writes_bytes(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(root=tmp_path)

    uri = await storage.put(key="abcdef1234", data=b"hello world")

    assert uri.startswith("file://")
    assert await storage.get(uri) == b"hello world"


async def test_put_is_idempotent_for_same_key_and_data(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(root=tmp_path)
    payload = b"content-addressable"

    first = await storage.put(key="deadbeef", data=payload)
    second = await storage.put(key="deadbeef", data=payload)

    assert first == second
    assert await storage.get(first) == payload


async def test_put_overwrites_when_key_is_reused(tmp_path: Path) -> None:
    """Re-putting under the same key replaces the blob.

    Callers are expected to use content-addressable keys (e.g. SHA-256), in
    which case overwriting with the same bytes is a no-op. If the bytes
    change, the caller has explicitly chosen to overwrite.
    """

    storage = LocalFilesystemStorage(root=tmp_path)

    uri = await storage.put(key="aabbcc", data=b"v1")
    await storage.put(key="aabbcc", data=b"v2")

    assert await storage.get(uri) == b"v2"


async def test_exists_reflects_presence(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(root=tmp_path)
    uri = await storage.put(key="ffeedd", data=b"x")

    assert await storage.exists(uri) is True
    assert await storage.exists("file:///nonexistent/path/abc") is False


async def test_get_raises_storage_error_for_missing_uri(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(root=tmp_path)

    with pytest.raises(StorageError):
        await storage.get(f"file://{tmp_path}/nope/missing")


async def test_invalid_keys_are_rejected(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(root=tmp_path)

    for bad in ("", "with/slash", "..escape", "../also-escape"):
        with pytest.raises(StorageError):
            await storage.put(key=bad, data=b"x")


async def test_unsupported_uri_scheme_is_rejected(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(root=tmp_path)

    with pytest.raises(StorageError):
        await storage.get("s3://bucket/key")


async def test_blobs_are_sharded_by_key_prefix(tmp_path: Path) -> None:
    """Sharding keeps any one directory from accumulating millions of files."""

    storage = LocalFilesystemStorage(root=tmp_path)

    uri = await storage.put(key="abXXXX", data=b"sharded")

    # The blob lives under <root>/ab/abXXXX
    expected = tmp_path.resolve() / "ab" / "abXXXX"
    assert expected.exists()
    assert uri == f"file://{expected}"
