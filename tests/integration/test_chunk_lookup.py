"""Integration tests for SqlChunkLookup against PostgreSQL."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

from sqlalchemy.orm import Session

from rag.db.models import ChunkOrm, DocumentOrm
from rag.db.repositories.chunk_lookup import SqlChunkLookup
from rag.domain.entities import Chunk, DocumentStatus
from rag.domain.interfaces import ChunkLookup

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _seed_document(session: Session) -> DocumentOrm:
    doc = DocumentOrm(
        id=uuid.uuid4(),
        filename="report.pdf",
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex,
        status=DocumentStatus.READY,
        doc_metadata={},
        created_at=BASE_TIME,
        updated_at=BASE_TIME,
    )
    session.add(doc)
    session.flush()
    return doc


def _seed_chunk(session: Session, doc_id: uuid.UUID, index: int, text: str) -> Chunk:
    cid = uuid.uuid4()
    orm = ChunkOrm(
        id=cid,
        document_id=doc_id,
        index=index,
        text=text,
        page=1,
        section="Intro",
        token_count=10,
        embedding_model_version="v1",
    )
    session.add(orm)
    session.flush()
    return Chunk(
        id=cid,
        document_id=doc_id,
        index=index,
        text=text,
        page=1,
        section="Intro",
        token_count=10,
        embedding_model_version="v1",
    )


def test_satisfies_protocol(session: Session) -> None:
    lookup: ChunkLookup = SqlChunkLookup(session)
    assert isinstance(lookup, ChunkLookup)


def test_get_many_preserves_order_and_skips_missing_ids(session: Session) -> None:
    doc = _seed_document(session)
    c1 = _seed_chunk(session, doc.id, 0, "First chunk")
    c2 = _seed_chunk(session, doc.id, 1, "Second chunk")
    c3 = _seed_chunk(session, doc.id, 2, "Third chunk")

    missing_id1 = uuid.uuid4()
    missing_id2 = uuid.uuid4()

    lookup = SqlChunkLookup(session)

    # Request order: c3, c1, missing_id1, c2, missing_id2
    results = lookup.get_many([c3.id, c1.id, missing_id1, c2.id, missing_id2])

    assert len(results) == 3
    assert [c.id for c in results] == [c3.id, c1.id, c2.id]
    assert results[0].text == "Third chunk"
    assert results[1].text == "First chunk"
    assert results[2].text == "Second chunk"


def test_get_many_empty_input(session: Session) -> None:
    lookup = SqlChunkLookup(session)
    assert lookup.get_many([]) == []


def test_sql_chunk_lookup_never_commits_or_rolls_back(session: Session) -> None:
    mock_session = MagicMock(spec=session)
    mock_session.scalars.return_value.all.return_value = []

    lookup = SqlChunkLookup(mock_session)
    lookup.get_many([uuid.uuid4()])

    mock_session.commit.assert_not_called()
    mock_session.rollback.assert_not_called()
