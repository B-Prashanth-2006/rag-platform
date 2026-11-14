"""Integration tests for SqlQueryLogRepository against a real Postgres instance."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from rag.db.repositories.query_logs import SqlQueryLogRepository
from rag.domain.entities import QueryLog
from rag.domain.errors import QueryLogNotFoundError
from rag.domain.interfaces import QueryLogRepository

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_log(*, minutes: int = 0, chunk_ids: list[uuid.UUID] | None = None) -> QueryLog:
    ts = BASE_TIME + timedelta(minutes=minutes)
    return QueryLog(
        id=uuid.uuid4(),
        query="What is RAG?",
        retrieved_chunk_ids=chunk_ids if chunk_ids is not None else [],
        latency_ms=42,
        answer="Retrieval Augmented Generation",
        feedback=None,
        created_at=ts,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(session: Session) -> SqlQueryLogRepository:
    return SqlQueryLogRepository(session)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_satisfies_protocol(session: Session) -> None:
    typed: QueryLogRepository = SqlQueryLogRepository(session)
    assert isinstance(typed, QueryLogRepository)


def test_add_roundtrip(repo: SqlQueryLogRepository, session: Session) -> None:
    chunk_ids = [uuid.uuid4(), uuid.uuid4()]
    log = make_log(chunk_ids=chunk_ids)
    repo.add(log)
    session.commit()
    session.expire_all()
    # list_recent returns the one we added
    results = repo.list_recent(limit=10)
    assert len(results) == 1
    result = results[0]
    assert result.id == log.id
    assert result.query == log.query
    assert result.latency_ms == log.latency_ms
    assert result.answer == log.answer
    assert result.feedback is None
    # UUID array round-trips correctly
    assert result.retrieved_chunk_ids == chunk_ids


def test_add_feedback_persists(repo: SqlQueryLogRepository, session: Session) -> None:
    log = make_log()
    repo.add(log)
    repo.add_feedback(log.id, "very helpful")
    session.commit()
    session.expire_all()
    result = repo.list_recent(limit=1)[0]
    assert result.feedback == "very helpful"


def test_list_recent_newest_first_with_limit(repo: SqlQueryLogRepository, session: Session) -> None:
    logs = [make_log(minutes=i) for i in range(5)]
    for log in logs:
        repo.add(log)
    # Newest first → minutes=4 is first
    recent = repo.list_recent(limit=3)
    assert len(recent) == 3
    assert recent[0].id == logs[4].id
    assert recent[1].id == logs[3].id
    assert recent[2].id == logs[2].id


def test_add_feedback_missing_raises(repo: SqlQueryLogRepository) -> None:
    with pytest.raises(QueryLogNotFoundError):
        repo.add_feedback(uuid.uuid4(), "feedback on nothing")
