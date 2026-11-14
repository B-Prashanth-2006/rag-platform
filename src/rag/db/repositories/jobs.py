"""PostgreSQL implementation of JobRepository.

The repository never commits. The caller owns the transaction.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from rag.db.models import IngestionJobOrm
from rag.domain.entities import IngestionJob, JobStatus
from rag.domain.errors import JobNotFoundError


def _to_entity(row: IngestionJobOrm) -> IngestionJob:
    return IngestionJob(
        id=row.id,
        document_id=row.document_id,
        status=JobStatus(row.status),
        error=row.error,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


def _to_orm(job: IngestionJob) -> IngestionJobOrm:
    return IngestionJobOrm(
        id=job.id,
        document_id=job.document_id,
        status=job.status,
        error=job.error,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


class SqlJobRepository:
    """JobRepository backed by a SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, job: IngestionJob) -> None:
        self._session.add(_to_orm(job))
        self._session.flush()

    def get(self, job_id: uuid.UUID) -> IngestionJob | None:
        row = self._session.get(IngestionJobOrm, job_id)
        return _to_entity(row) if row is not None else None

    def _get_or_raise(self, job_id: uuid.UUID) -> IngestionJobOrm:
        row = self._session.get(IngestionJobOrm, job_id)
        if row is None:
            raise JobNotFoundError(job_id)
        return row

    def mark_running(self, job_id: uuid.UUID, started_at: datetime) -> None:
        row = self._get_or_raise(job_id)
        row.status = JobStatus.RUNNING
        row.started_at = started_at
        self._session.flush()

    def mark_succeeded(self, job_id: uuid.UUID, finished_at: datetime) -> None:
        row = self._get_or_raise(job_id)
        row.status = JobStatus.SUCCEEDED
        row.finished_at = finished_at
        self._session.flush()

    def mark_failed(self, job_id: uuid.UUID, error: str, finished_at: datetime) -> None:
        row = self._get_or_raise(job_id)
        row.status = JobStatus.FAILED
        row.error = error
        row.finished_at = finished_at
        self._session.flush()
