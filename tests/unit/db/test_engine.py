"""Unit tests for the sync engine and session factory — no live DB required."""

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from rag.db.engine import SessionLocal, engine


def test_engine_is_sync_engine() -> None:
    assert isinstance(engine, Engine)


def test_session_local_is_sessionmaker() -> None:
    assert isinstance(SessionLocal, sessionmaker)


def test_engine_url_uses_postgresql() -> None:
    url = str(engine.url)
    assert url.startswith("postgresql")


def test_engine_url_uses_psycopg_driver() -> None:
    url = str(engine.url)
    assert "psycopg" in url


def test_session_maker_class_is_session() -> None:
    assert issubclass(SessionLocal.class_, Session)
