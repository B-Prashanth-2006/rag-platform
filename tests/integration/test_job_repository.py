"""Integration tests for SqlJobRepository against a real Postgres instance."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from rag.db.models import DocumentOrm
from rag.db.repositories.jobs import SqlJobRepository
from rag.domain.entities import DocumentStatus, IngestionJob, JobStatus
from rag.domain.errors import JobNotFoundError
from rag.domain.interfaces import JobRepository

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)
RUN_TIME = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
DONE_TIME = datetime(2026, 1, 1, 0, 2, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_doc(session: Session) -> DocumentOrm:
    doc = DocumentOrm(
        id=uuid.uuid4(),
        filename="test.pdf",
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex,
        status=DocumentStatus.PENDING,
        doc_metadata={},
        created_at=BASE_TIME,
        updated_at=BASE_TIME,
    )
    session.add(doc)
    session.flush()
    return doc


def make_job(doc_id: uuid.UUID) -> IngestionJob:
    return IngestionJob(
        id=uuid.uuid4(),
        document_id=doc_id,
        status=JobStatus.QUEUED,
        error=None,
        started_at=None,
        finished_at=None,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(session: Session) -> SqlJobRepository:
    return SqlJobRepository(session)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_satisfies_protocol(session: Session) -> None:
    typed: JobRepository = SqlJobRepository(session)
    assert isinstance(typed, JobRepository)


def test_create_and_get(repo: SqlJobRepository, session: Session) -> None:
    doc = make_doc(session)
    job = make_job(doc.id)
    repo.create(job)
    session.commit()
    session.expire_all()
    result = repo.get(job.id)
    assert result is not None
    assert result.id == job.id
    assert result.status == JobStatus.QUEUED
    assert result.error is None
    assert result.started_at is None
    assert result.finished_at is None


def test_full_success_transition(repo: SqlJobRepository, session: Session) -> None:
    doc = make_doc(session)
    job = make_job(doc.id)
    repo.create(job)

    repo.mark_running(job.id, RUN_TIME)
    mid = repo.get(job.id)
    assert mid is not None
    assert mid.status == JobStatus.RUNNING
    assert mid.started_at == RUN_TIME

    repo.mark_succeeded(job.id, DONE_TIME)
    session.commit()
    session.expire_all()
    done = repo.get(job.id)
    assert done is not None
    assert done.status == JobStatus.SUCCEEDED
    assert done.finished_at == DONE_TIME
    assert done.error is None


def test_full_failure_transition(repo: SqlJobRepository, session: Session) -> None:
    doc = make_doc(session)
    job = make_job(doc.id)
    repo.create(job)

    repo.mark_running(job.id, RUN_TIME)
    repo.mark_failed(job.id, "timeout reading source", DONE_TIME)
    session.commit()
    session.expire_all()
    done = repo.get(job.id)
    assert done is not None
    assert done.status == JobStatus.FAILED
    assert done.error == "timeout reading source"
    assert done.finished_at == DONE_TIME


def test_missing_job_raises(repo: SqlJobRepository, session: Session) -> None:
    with pytest.raises(JobNotFoundError):
        repo.mark_running(uuid.uuid4(), RUN_TIME)

