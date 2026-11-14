"""Integration tests for SqlChunkRepository against a real Postgres instance."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from rag.db.models import DocumentOrm
from rag.db.repositories.chunks import SqlChunkRepository
from rag.domain.entities import Chunk, DocumentStatus
from rag.domain.errors import DuplicateChunkIndexError
from rag.domain.interfaces import ChunkRepository

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_doc(session: Session) -> DocumentOrm:
    """Insert a minimal DocumentOrm and return it (unflushed — caller flushes)."""
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


def make_chunk(doc_id: uuid.UUID, index: int, version: str | None = "v1") -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=doc_id,
        index=index,
        text=f"chunk text {index}",
        page=None,
        section=None,
        token_count=10,
        embedding_model_version=version,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(session: Session) -> SqlChunkRepository:
    return SqlChunkRepository(session)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_satisfies_protocol(session: Session) -> None:
    typed: ChunkRepository = SqlChunkRepository(session)
    assert isinstance(typed, ChunkRepository)


def test_add_many_roundtrip(repo: SqlChunkRepository, session: Session) -> None:
    doc = make_doc(session)
    chunks = [make_chunk(doc.id, i) for i in range(3)]
    repo.add_many(chunks)
    session.commit()
    session.expire_all()
    result = repo.list_by_document(doc.id)
    assert len(result) == 3
    assert {c.id for c in result} == {c.id for c in chunks}


def test_list_ordered_by_index(repo: SqlChunkRepository, session: Session) -> None:
    doc = make_doc(session)
    # Insert in reverse order to prove ordering is by index not insertion
    repo.add_many([make_chunk(doc.id, i) for i in [2, 0, 1]])
    result = repo.list_by_document(doc.id)
    assert [c.index for c in result] == [0, 1, 2]


def test_duplicate_index_rejected(repo: SqlChunkRepository, session: Session) -> None:
    doc = make_doc(session)
    repo.add_many([make_chunk(doc.id, 0)])
    with pytest.raises(DuplicateChunkIndexError):
        repo.add_many([make_chunk(doc.id, 0)])


def test_duplicate_does_not_discard_work(repo: SqlChunkRepository, session: Session) -> None:
    doc = make_doc(session)
    first = make_chunk(doc.id, 0)
    repo.add_many([first])
    with pytest.raises(DuplicateChunkIndexError):
        repo.add_many([make_chunk(doc.id, 0)])  # duplicate index
    # The savepoint was rolled back but the first add_many survives
    result = repo.list_by_document(doc.id)
    assert any(c.id == first.id for c in result)


def test_delete_by_document(repo: SqlChunkRepository, session: Session) -> None:
    doc = make_doc(session)
    repo.add_many([make_chunk(doc.id, i) for i in range(3)])
    repo.delete_by_document(doc.id)
    assert repo.list_by_document(doc.id) == []


def test_count_by_model_version(repo: SqlChunkRepository, session: Session) -> None:
    doc = make_doc(session)
    repo.add_many(
        [
            make_chunk(doc.id, 0, version="v1"),
            make_chunk(doc.id, 1, version="v1"),
            make_chunk(doc.id, 2, version="v2"),
            make_chunk(doc.id, 3, version=None),  # NULL — must not be counted
        ]
    )
    assert repo.count_by_model_version("v1") == 2
    assert repo.count_by_model_version("v2") == 1
    assert repo.count_by_model_version("v3") == 0


def test_add_many_empty_list_is_noop(repo: SqlChunkRepository, session: Session) -> None:
    doc = make_doc(session)
    repo.add_many([])  # must not raise
    assert repo.list_by_document(doc.id) == []


def test_parent_delete_cascades(repo: SqlChunkRepository, session: Session) -> None:
    doc = make_doc(session)
    repo.add_many([make_chunk(doc.id, i) for i in range(2)])
    session.flush()
    session.delete(doc)
    session.flush()
    assert repo.list_by_document(doc.id) == []
