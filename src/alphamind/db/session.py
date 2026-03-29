"""Async SQLAlchemy engine and session management.

The engine and session factory are lazily instantiated module-level
singletons. Tests that need to swap the DSN do so by clearing the LRU caches
on :func:`get_engine` and :func:`get_session_factory`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from alphamind.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Return the process-wide async engine."""

    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        future=True,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide async session factory."""

    return async_sessionmaker(
        bind=get_engine(),
        expire_on_commit=False,
        class_=AsyncSession,
    )


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Yield a session that commits on success and rolls back on error.

    Prefer this over calling ``get_session_factory()`` directly at call sites
    that represent a unit of work.
    """

    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()


async def dispose_engine() -> None:
    """Dispose of the engine and clear cached factories.

    Used by tests and on graceful shutdown.
    """

    engine = get_engine()
    await engine.dispose()
    get_engine.cache_clear()
    get_session_factory.cache_clear()


__all__ = ["dispose_engine", "get_engine", "get_session_factory", "session_scope"]
