"""Integration tests for rag.db.session.get_session.

Uses the testcontainers engine fixture (real Postgres) so SQLite is never involved.
SessionLocal is monkeypatched to bind to the testcontainer engine, which lets us
exercise the full commit / rollback paths in get_session without touching the
app's default database URL.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

import rag.db.session as session_module
from rag.db.session import get_session


@pytest.fixture
def patched_session_local(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace SessionLocal in rag.db.session with one bound to the testcontainer engine."""
    factory: sessionmaker[Session] = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(session_module, "SessionLocal", factory)


def test_get_session_commits_on_success(
    engine: Engine, patched_session_local: None
) -> None:
    """Data written inside the context manager is visible after it exits."""
    with get_session() as session:
        session.execute(
            text(
                "INSERT INTO query_logs(id, query, retrieved_chunk_ids, latency_ms, created_at) "
                "VALUES (gen_random_uuid(), 'test', '{}', 1, now())"
            )
        )

    with engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM query_logs")).scalar()
    assert count == 1


def test_get_session_rolls_back_on_error(
    engine: Engine, patched_session_local: None
) -> None:
    """An exception inside the context manager causes a rollback."""
    # Snapshot count before we start, so a row left by a previous test doesn't
    # cause a false failure. The rollback must not add any new rows.
    with engine.connect() as conn:
        before = conn.execute(text("SELECT count(*) FROM query_logs")).scalar() or 0

    try:
        with get_session() as session:
            session.execute(
                text(
                    "INSERT INTO query_logs"
                    "(id, query, retrieved_chunk_ids, latency_ms, created_at) "
                    "VALUES (gen_random_uuid(), 'rollback-test', '{}', 1, now())"
                )
            )
            raise RuntimeError("deliberate")
    except RuntimeError:
        pass  # expected

    with engine.connect() as conn:
        after = conn.execute(text("SELECT count(*) FROM query_logs")).scalar() or 0
    assert after == before  # the rolled-back row was not committed


def test_get_session_closes_session(patched_session_local: None) -> None:
    """The session is not in a transaction after the context exits."""
    with get_session() as session:
        captured = session
    # After close(), the session has no active transaction.
    assert not captured.in_transaction()
