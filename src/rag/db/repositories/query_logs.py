"""PostgreSQL implementation of QueryLogRepository.

The repository never commits. The caller owns the transaction.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag.db.models import QueryLogOrm
from rag.domain.entities import QueryLog
from rag.domain.errors import QueryLogNotFoundError


def _to_entity(row: QueryLogOrm) -> QueryLog:
    return QueryLog(
        id=row.id,
        query=row.query,
        retrieved_chunk_ids=list(row.retrieved_chunk_ids),
        latency_ms=row.latency_ms,
        answer=row.answer,
        feedback=row.feedback,
        created_at=row.created_at,
    )


def _to_orm(log: QueryLog) -> QueryLogOrm:
    return QueryLogOrm(
        id=log.id,
        query=log.query,
        retrieved_chunk_ids=list(log.retrieved_chunk_ids),
        latency_ms=log.latency_ms,
        answer=log.answer,
        feedback=log.feedback,
        created_at=log.created_at,
    )


class SqlQueryLogRepository:
    """QueryLogRepository backed by a SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, log: QueryLog) -> None:
        self._session.add(_to_orm(log))
        self._session.flush()

    def add_feedback(self, log_id: uuid.UUID, feedback: str) -> None:
        row = self._session.get(QueryLogOrm, log_id)
        if row is None:
            raise QueryLogNotFoundError(log_id)
        row.feedback = feedback
        self._session.flush()

    def list_recent(self, limit: int) -> list[QueryLog]:
        stmt = (
            select(QueryLogOrm)
            .order_by(QueryLogOrm.created_at.desc(), QueryLogOrm.id.desc())
            .limit(limit)
        )
        return [_to_entity(row) for row in self._session.scalars(stmt).all()]

