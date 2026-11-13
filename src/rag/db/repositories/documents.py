"""PostgreSQL implementation of DocumentRepository.

The repository never commits. The caller (get_session or a unit of work) owns
the transaction. flush() is used so constraint violations surface immediately.
"""

from __future__ import annotations

import builtins
import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from rag.db.models import DocumentOrm
from rag.domain.entities import Document, DocumentStatus
from rag.domain.errors import DocumentNotFoundError, DuplicateContentHashError


def _to_entity(row: DocumentOrm) -> Document:
    return Document(
        id=row.id,
        filename=row.filename,
        content_hash=row.content_hash,
        status=DocumentStatus(row.status),
        metadata=dict(row.doc_metadata),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_orm(doc: Document) -> DocumentOrm:
    return DocumentOrm(
        id=doc.id,
        filename=doc.filename,
        content_hash=doc.content_hash,
        status=doc.status,
        doc_metadata=dict(doc.metadata),
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


class SqlDocumentRepository:
    """DocumentRepository backed by a SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, doc: Document) -> None:
        try:
            with self._session.begin_nested():  # SAVEPOINT: only this insert is undone on failure
                self._session.add(_to_orm(doc))
        except IntegrityError as exc:
            if "content_hash" in str(exc.orig):
                raise DuplicateContentHashError(doc.content_hash) from exc
            raise

    def get(self, doc_id: uuid.UUID) -> Document | None:
        row = self._session.get(DocumentOrm, doc_id)
        return _to_entity(row) if row is not None else None

    def get_by_hash(self, content_hash: str) -> Document | None:
        stmt = select(DocumentOrm).where(DocumentOrm.content_hash == content_hash)
        row = self._session.scalars(stmt).one_or_none()
        return _to_entity(row) if row is not None else None

    def update_status(self, doc_id: uuid.UUID, status: DocumentStatus) -> None:
        row = self._session.get(DocumentOrm, doc_id)
        if row is None:
            raise DocumentNotFoundError(doc_id)
        row.status = status
        self._session.flush()

    def delete(self, doc_id: uuid.UUID) -> None:
        """Delete a document and its chunks and jobs. Deleting a missing id is a no-op."""
        row = self._session.get(DocumentOrm, doc_id)
        if row is not None:
            self._session.delete(row)
            self._session.flush()

    def list(
        self,
        limit: int,
        offset: int,
        status: DocumentStatus | None = None,
    ) -> builtins.list[Document]:
        stmt = select(DocumentOrm)
        if status is not None:
            stmt = stmt.where(DocumentOrm.status == status)
        stmt = stmt.order_by(DocumentOrm.created_at, DocumentOrm.id).limit(limit).offset(offset)
        return [_to_entity(row) for row in self._session.scalars(stmt).all()]
