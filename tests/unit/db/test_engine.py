"""Unit tests for the async engine and session factory — no live DB required."""

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from rag.db.engine import AsyncSessionLocal, engine


def test_engine_is_async_engine() -> None:
    assert isinstance(engine, AsyncEngine)


def test_session_local_is_async_sessionmaker() -> None:
    assert isinstance(AsyncSessionLocal, async_sessionmaker)


def test_engine_url_uses_postgresql() -> None:
    url = str(engine.url)
    assert url.startswith("postgresql")


def test_session_maker_class_is_async_session() -> None:
    assert AsyncSessionLocal.class_ is AsyncSession

