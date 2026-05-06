"""Verify local dependencies: Postgres (with pgvector) and Redis.

Run with:

    make healthcheck

The script fails fast on any unreachable service and exits non-zero so it can
gate CI jobs or local scripts that assume a working environment.
"""

from __future__ import annotations

import asyncio
import sys

import redis.asyncio as redis
from sqlalchemy import text

from alphamind.config import get_settings
from alphamind.db.session import dispose_engine, get_engine


async def check_postgres() -> None:
    engine = get_engine()
    async with engine.connect() as conn:
        version = (await conn.execute(text("SELECT version()"))).scalar_one()
        vector_ext = (
            await conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
        ).scalar_one_or_none()

    short = str(version).split(" on ", maxsplit=1)[0]
    print(f"  postgres  ok  ({short})")
    if vector_ext != "vector":
        raise RuntimeError(
            "pgvector extension is not installed. Run `make migrate` to apply the baseline."
        )
    print("  pgvector  ok")


async def check_redis() -> None:
    settings = get_settings()
    client = redis.from_url(settings.redis_url)
    try:
        if not await client.ping():
            raise RuntimeError("redis ping returned falsy")
        info = await client.info(section="server")
        version = info.get("redis_version", "unknown")
    finally:
        await client.aclose()
    print(f"  redis     ok  (v{version})")


async def main() -> None:
    print("alphamind healthcheck")
    try:
        await check_postgres()
        await check_redis()
    finally:
        await dispose_engine()
    print("all services healthy")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"healthcheck failed: {exc}", file=sys.stderr)
        sys.exit(1)
