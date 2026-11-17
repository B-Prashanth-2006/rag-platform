"""Integration tests for SqlDocumentLookup against PostgreSQL."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from rag.db.models import DocumentOrm
from rag.db.repositories.document_lookup import SqlDocumentLookup
from rag.domain.entities import DocumentStatus
from rag.domain.interfaces import DocumentLookup


def test_get_names_resolves_existing_documents_and_skips_missing_ids(session: Session) -> None:
    document_id = uuid.uuid4()
    missing_id = uuid.uuid4()
    session.add(
        DocumentOrm(
            id=document_id,
            filename="manual.pdf",
            content_hash=uuid.uuid4().hex + uuid.uuid4().hex,
            status=DocumentStatus.READY,
            doc_metadata={},
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    session.flush()
    lookup: DocumentLookup = SqlDocumentLookup(session)

    assert lookup.get_names([missing_id, document_id]) == {document_id: "manual.pdf"}
    assert session.in_transaction()
    assert isinstance(lookup, DocumentLookup)
