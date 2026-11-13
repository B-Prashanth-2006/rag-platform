from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag.db.models import ChunkOrm
from rag.db.repositories.documents import SqlDocumentRepository
from rag.domain.entities import Document, DocumentStatus
from rag.domain.errors import DocumentNotFoundError, DuplicateContentHashError
from rag.domain.interfaces import DocumentRepository

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def make_doc(
    *,
    content_hash: str | None = None,
    status: DocumentStatus = DocumentStatus.PENDING,
    minutes: int = 0,
    filename: str = "a.pdf",
) -> Document:
    ts = BASE_TIME + timedelta(minutes=minutes)
    return Document(
        id=uuid.uuid4(),
        filename=filename,
        content_hash=content_hash or (uuid.uuid4().hex + uuid.uuid4().hex),
        status=status,
        metadata={"pages": 3, "tags": ["x"]},
        created_at=ts,
        updated_at=ts,
    )


@pytest.fixture
def repo(session: Session) -> SqlDocumentRepository:
    return SqlDocumentRepository(session)


def test_satisfies_protocol(session: Session) -> None:
    typed: DocumentRepository = SqlDocumentRepository(session)  # mypy checks the contract
    assert isinstance(typed, DocumentRepository)


def test_add_and_get_roundtrip(repo: SqlDocumentRepository, session: Session) -> None:
    doc = make_doc()
    repo.add(doc)
    session.commit()
    session.expire_all()  # force a real read from the database
    assert repo.get(doc.id) == doc


def test_get_missing_returns_none(repo: SqlDocumentRepository) -> None:
    assert repo.get(uuid.uuid4()) is None


def test_get_by_hash(repo: SqlDocumentRepository) -> None:
    doc = make_doc(content_hash="a" * 64)
    repo.add(doc)
    found = repo.get_by_hash("a" * 64)
    assert found is not None
    assert found.id == doc.id
    assert repo.get_by_hash("b" * 64) is None


def test_duplicate_content_hash_is_rejected(repo: SqlDocumentRepository) -> None:
    repo.add(make_doc(content_hash="c" * 64, filename="one.pdf"))
    with pytest.raises(DuplicateContentHashError):
        repo.add(make_doc(content_hash="c" * 64, filename="two.pdf"))


def test_list_paginates_in_creation_order(repo: SqlDocumentRepository) -> None:
    docs = [make_doc(minutes=i) for i in range(5)]
    for d in docs:
        repo.add(d)
    page1 = repo.list(limit=2, offset=0)
    page2 = repo.list(limit=2, offset=2)
    page3 = repo.list(limit=2, offset=4)
    assert [d.id for d in page1] == [docs[0].id, docs[1].id]
    assert [d.id for d in page2] == [docs[2].id, docs[3].id]
    assert [d.id for d in page3] == [docs[4].id]


def test_list_filters_by_status(repo: SqlDocumentRepository) -> None:
    ready = make_doc(status=DocumentStatus.READY, minutes=1)
    repo.add(make_doc(status=DocumentStatus.PENDING, minutes=0))
    repo.add(ready)
    result = repo.list(limit=10, offset=0, status=DocumentStatus.READY)
    assert [d.id for d in result] == [ready.id]


def test_update_status(repo: SqlDocumentRepository, session: Session) -> None:
    doc = make_doc()
    repo.add(doc)
    repo.update_status(doc.id, DocumentStatus.READY)
    session.commit()
    session.expire_all()
    fetched = repo.get(doc.id)
    assert fetched is not None
    assert fetched.status is DocumentStatus.READY


def test_update_status_missing_raises(repo: SqlDocumentRepository) -> None:
    with pytest.raises(DocumentNotFoundError):
        repo.update_status(uuid.uuid4(), DocumentStatus.READY)


def test_delete_cascades_to_chunks(repo: SqlDocumentRepository, session: Session) -> None:
    doc = make_doc()
    repo.add(doc)
    session.add(ChunkOrm(id=uuid.uuid4(), document_id=doc.id, index=0, text="hello", token_count=1))
    session.commit()

    repo.delete(doc.id)
    session.commit()

    assert repo.get(doc.id) is None
    remaining = session.scalar(select(func.count()).select_from(ChunkOrm))
    assert remaining == 0


def test_delete_missing_is_noop(repo: SqlDocumentRepository) -> None:
    repo.delete(uuid.uuid4())  # must not raise


def test_rollback_discards_uncommitted_add(repo: SqlDocumentRepository, session: Session) -> None:
    doc = make_doc()
    repo.add(doc)
    session.rollback()
    assert repo.get(doc.id) is None


def test_duplicate_does_not_discard_earlier_work(
    repo: SqlDocumentRepository, session: Session
) -> None:
    first = make_doc(content_hash="d" * 64)
    repo.add(first)
    with pytest.raises(DuplicateContentHashError):
        repo.add(make_doc(content_hash="d" * 64))
    # the session is still usable and the first insert survived
    assert repo.get(first.id) is not None
