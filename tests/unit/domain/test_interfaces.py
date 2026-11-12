"""Unit tests for domain repository Protocols — written first (TDD red phase)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from rag.domain.entities import Chunk, Document, DocumentStatus, IngestionJob, JobStatus, QueryLog
from rag.domain.interfaces import (
    ChunkRepository,
    DocumentRepository,
    IngestionJobRepository,
    QueryLogRepository,
)

# ---------------------------------------------------------------------------
# Minimal in-memory stubs that satisfy each Protocol
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _doc() -> Document:
    return Document(
        id=uuid.uuid4(),
        filename="f.pdf",
        content_hash="hash1",
        status=DocumentStatus.PENDING,
        metadata={},
        created_at=_now(),
        updated_at=_now(),
    )


def _chunk(doc_id: uuid.UUID | None = None) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=doc_id or uuid.uuid4(),
        index=0,
        text="text",
        page=None,
        section=None,
        token_count=5,
        embedding_model_version=None,
    )


def _job(doc_id: uuid.UUID | None = None) -> IngestionJob:
    return IngestionJob(
        id=uuid.uuid4(),
        document_id=doc_id or uuid.uuid4(),
        status=JobStatus.QUEUED,
        error=None,
        started_at=None,
        finished_at=None,
    )


def _qlog() -> QueryLog:
    return QueryLog(
        id=uuid.uuid4(),
        query="q",
        retrieved_chunk_ids=[],
        latency_ms=1,
        answer=None,
        feedback=None,
        created_at=_now(),
    )


# ---------------------------------------------------------------------------
# Stub implementations
# ---------------------------------------------------------------------------


class InMemoryDocumentRepository:
    def __init__(self) -> None:
        self._store: dict[uuid.UUID, Document] = {}

    async def add(self, doc: Document) -> None:
        self._store[doc.id] = doc

    async def get(self, doc_id: uuid.UUID) -> Document | None:
        return self._store.get(doc_id)

    async def get_by_hash(self, content_hash: str) -> Document | None:
        return next((d for d in self._store.values() if d.content_hash == content_hash), None)

    async def update_status(self, doc_id: uuid.UUID, status: DocumentStatus) -> None:
        if doc_id in self._store:
            existing = self._store[doc_id]
            self._store[doc_id] = Document(
                id=existing.id,
                filename=existing.filename,
                content_hash=existing.content_hash,
                status=status,
                metadata=existing.metadata,
                created_at=existing.created_at,
                updated_at=_now(),
            )

    async def list_all(self, limit: int, offset: int) -> list[Document]:
        items = list(self._store.values())
        return items[offset : offset + limit]


class InMemoryChunkRepository:
    def __init__(self) -> None:
        self._store: list[Chunk] = []

    async def add_many(self, chunks: list[Chunk]) -> None:
        self._store.extend(chunks)

    async def get_by_document(self, document_id: uuid.UUID) -> list[Chunk]:
        return [c for c in self._store if c.document_id == document_id]

    async def delete_by_document(self, document_id: uuid.UUID) -> None:
        self._store = [c for c in self._store if c.document_id != document_id]


class InMemoryIngestionJobRepository:
    def __init__(self) -> None:
        self._store: dict[uuid.UUID, IngestionJob] = {}

    async def add(self, job: IngestionJob) -> None:
        self._store[job.id] = job

    async def get(self, job_id: uuid.UUID) -> IngestionJob | None:
        return self._store.get(job_id)

    async def update(self, job: IngestionJob) -> None:
        self._store[job.id] = job


class InMemoryQueryLogRepository:
    def __init__(self) -> None:
        self._store: dict[uuid.UUID, QueryLog] = {}

    async def add(self, log: QueryLog) -> None:
        self._store[log.id] = log

    async def get(self, log_id: uuid.UUID) -> QueryLog | None:
        return self._store.get(log_id)


# ---------------------------------------------------------------------------
# Classes that are MISSING methods — should NOT satisfy Protocol
# ---------------------------------------------------------------------------


class IncompleteDocumentRepository:
    """Missing update_status and list_all."""

    async def add(self, doc: Document) -> None: ...
    async def get(self, doc_id: uuid.UUID) -> Document | None: ...
    async def get_by_hash(self, content_hash: str) -> Document | None: ...


# ---------------------------------------------------------------------------
# Tests: Protocol structural subtyping
# ---------------------------------------------------------------------------


class TestDocumentRepositoryProtocol:
    def test_valid_stub_satisfies_protocol(self) -> None:
        repo: Any = InMemoryDocumentRepository()
        assert isinstance(repo, DocumentRepository)

    def test_incomplete_stub_does_not_satisfy_protocol(self) -> None:
        repo: Any = IncompleteDocumentRepository()
        assert not isinstance(repo, DocumentRepository)

    def test_plain_object_does_not_satisfy_protocol(self) -> None:
        assert not isinstance(object(), DocumentRepository)


class TestChunkRepositoryProtocol:
    def test_valid_stub_satisfies_protocol(self) -> None:
        repo: Any = InMemoryChunkRepository()
        assert isinstance(repo, ChunkRepository)


class TestIngestionJobRepositoryProtocol:
    def test_valid_stub_satisfies_protocol(self) -> None:
        repo: Any = InMemoryIngestionJobRepository()
        assert isinstance(repo, IngestionJobRepository)


class TestQueryLogRepositoryProtocol:
    def test_valid_stub_satisfies_protocol(self) -> None:
        repo: Any = InMemoryQueryLogRepository()
        assert isinstance(repo, QueryLogRepository)


# ---------------------------------------------------------------------------
# Tests: basic behavioural correctness of the in-memory stubs
# (also validates the Protocol signatures are usable end-to-end)
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402  (import after Protocol tests for clarity)


def run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


class TestDocumentRepositoryBehaviour:
    def test_add_and_get(self) -> None:
        repo = InMemoryDocumentRepository()
        doc = _doc()
        run(repo.add(doc))
        result = run(repo.get(doc.id))
        assert result == doc

    def test_get_missing_returns_none(self) -> None:
        repo = InMemoryDocumentRepository()
        assert run(repo.get(uuid.uuid4())) is None

    def test_get_by_hash(self) -> None:
        repo = InMemoryDocumentRepository()
        doc = _doc()
        run(repo.add(doc))
        result = run(repo.get_by_hash(doc.content_hash))
        assert result is not None
        assert result.id == doc.id

    def test_update_status(self) -> None:
        repo = InMemoryDocumentRepository()
        doc = _doc()
        run(repo.add(doc))
        run(repo.update_status(doc.id, DocumentStatus.READY))
        updated = run(repo.get(doc.id))
        assert updated is not None
        assert updated.status == DocumentStatus.READY

    def test_list_all_pagination(self) -> None:
        repo = InMemoryDocumentRepository()
        for _ in range(5):
            run(repo.add(_doc()))
        page = run(repo.list_all(limit=2, offset=0))
        assert len(page) == 2
        page2 = run(repo.list_all(limit=2, offset=2))
        assert len(page2) == 2


class TestChunkRepositoryBehaviour:
    def test_add_many_and_get_by_document(self) -> None:
        repo = InMemoryChunkRepository()
        did = uuid.uuid4()
        chunks = [_chunk(did) for _ in range(3)]
        run(repo.add_many(chunks))
        result = run(repo.get_by_document(did))
        assert len(result) == 3

    def test_delete_by_document(self) -> None:
        repo = InMemoryChunkRepository()
        did = uuid.uuid4()
        run(repo.add_many([_chunk(did), _chunk(did)]))
        run(repo.delete_by_document(did))
        assert run(repo.get_by_document(did)) == []


class TestIngestionJobRepositoryBehaviour:
    def test_add_and_get(self) -> None:
        repo = InMemoryIngestionJobRepository()
        job = _job()
        run(repo.add(job))
        result = run(repo.get(job.id))
        assert result == job

    def test_update(self) -> None:
        repo = InMemoryIngestionJobRepository()
        job = _job()
        run(repo.add(job))
        updated_job = IngestionJob(
            id=job.id,
            document_id=job.document_id,
            status=JobStatus.SUCCEEDED,
            error=None,
            started_at=job.started_at,
            finished_at=_now(),
        )
        run(repo.update(updated_job))
        result = run(repo.get(job.id))
        assert result is not None
        assert result.status == JobStatus.SUCCEEDED


class TestQueryLogRepositoryBehaviour:
    def test_add_and_get(self) -> None:
        repo = InMemoryQueryLogRepository()
        log = _qlog()
        run(repo.add(log))
        result = run(repo.get(log.id))
        assert result == log

    def test_get_missing_returns_none(self) -> None:
        repo = InMemoryQueryLogRepository()
        assert run(repo.get(uuid.uuid4())) is None
