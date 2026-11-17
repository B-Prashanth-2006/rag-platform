"""PostgreSQL implementation of DocumentLookup."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag.db.models import DocumentOrm


class SqlDocumentLookup:
    """Resolve document IDs to filenames using a SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_names(self, ids: list[uuid.UUID]) -> dict[uuid.UUID, str]:
        """Return filenames for found documents without changing transaction state."""
        if not ids:
            return {}
        stmt = select(DocumentOrm.id, DocumentOrm.filename).where(DocumentOrm.id.in_(ids))
        return dict(self._session.execute(stmt).all())
