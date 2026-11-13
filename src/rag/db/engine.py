"""Async SQLAlchemy engine and session-maker factory.

Both ``engine`` and ``AsyncSessionLocal`` are module-level singletons created
once at import time.  All settings come from :func:`rag.config.get_settings`;
no URLs or pool parameters are hard-coded here.
"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rag.config import get_settings


def _make_engine() -> AsyncEngine:
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        echo=(settings.app_env == "dev"),
        future=True,
    )


engine: AsyncEngine = _make_engine()

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

