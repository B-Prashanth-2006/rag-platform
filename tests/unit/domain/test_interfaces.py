"""Unit tests for domain repository Protocols — written first (TDD red phase)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from rag.domain.entities import Chunk, Document, DocumentStatus, IngestionJob, JobStatus, QueryLog
from rag.domain.interfaces import (
    ChunkRepository,
    DocumentRepository,
    JobRepository,
    QueryLogRepository,
)

# ---------------------------------------------------------------------------
# Helpers
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

    def add(self, doc: Document) -> None:
        self._store[doc.id] = doc

    def get(self, doc_id: uuid.UUID) -> Document | None:
        return self._store.get(doc_id)

    def get_by_hash(self, content_hash: str) -> Document | None:
        return next((d for d in self._store.values() if d.content_hash == content_hash), None)

    def list(
        self,
        limit: int,
        offset: int,
        status: DocumentStatus | None = None,
    ) -> list[Document]:
        items = list(self._store.values())
        if status is not None:
            items = [d for d in items if d.status == status]
        return items[offset : offset + limit]

    def update_status(self, doc_id: uuid.UUID, status: DocumentStatus) -> None:
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

    def delete(self, doc_id: uuid.UUID) -> None:
        self._store.pop(doc_id, None)


class InMemoryChunkRepository:
    def __init__(self) -> None:
        self._store: list[Chunk] = []

    def add_many(self, chunks: list[Chunk]) -> None:
        self._store.extend(chunks)

    def list_by_document(self, document_id: uuid.UUID) -> list[Chunk]:
        return [c for c in self._store if c.document_id == document_id]

    def delete_by_document(self, document_id: uuid.UUID) -> None:
        self._store = [c for c in self._store if c.document_id != document_id]

    def count_by_model_version(self, version: str) -> int:
        return sum(1 for c in self._store if c.embedding_model_version == version)


class InMemoryJobRepository:
    def __init__(self) -> None:
        self._store: dict[uuid.UUID, IngestionJob] = {}

    def create(self, job: IngestionJob) -> None:
        self._store[job.id] = job

    def get(self, job_id: uuid.UUID) -> IngestionJob | None:
        return self._store.get(job_id)

    def mark_running(self, job_id: uuid.UUID, started_at: datetime) -> None:
        job = self._store.get(job_id)
        if job is not None:
            self._store[job_id] = IngestionJob(
                id=job.id,
                document_id=job.document_id,
                status=JobStatus.RUNNING,
                error=job.error,
                started_at=started_at,
                finished_at=job.finished_at,
            )

    def mark_succeeded(self, job_id: uuid.UUID, finished_at: datetime) -> None:
        job = self._store.get(job_id)
        if job is not None:
            self._store[job_id] = IngestionJob(
                id=job.id,
                document_id=job.document_id,
                status=JobStatus.SUCCEEDED,
                error=job.error,
                started_at=job.started_at,
                finished_at=finished_at,
            )

    def mark_failed(self, job_id: uuid.UUID, error: str, finished_at: datetime) -> None:
        job = self._store.get(job_id)
        if job is not None:
            self._store[job_id] = IngestionJob(
                id=job.id,
                document_id=job.document_id,
                status=JobStatus.FAILED,
                error=error,
                started_at=job.started_at,
                finished_at=finished_at,
            )


class InMemoryQueryLogRepository:
    def __init__(self) -> None:
        self._store: dict[uuid.UUID, QueryLog] = {}

    def add(self, log: QueryLog) -> None:
        self._store[log.id] = log

    def add_feedback(self, log_id: uuid.UUID, feedback: str) -> None:
        log = self._store.get(log_id)
        if log is not None:
            self._store[log_id] = QueryLog(
                id=log.id,
                query=log.query,
                retrieved_chunk_ids=log.retrieved_chunk_ids,
                latency_ms=log.latency_ms,
                answer=log.answer,
                feedback=feedback,
                created_at=log.created_at,
            )

    def list_recent(self, limit: int) -> list[QueryLog]:
        items = sorted(self._store.values(), key=lambda q: q.created_at, reverse=True)
        return items[:limit]


# ---------------------------------------------------------------------------
# Classes that are MISSING methods — should NOT satisfy Protocol
# ---------------------------------------------------------------------------


class IncompleteDocumentRepository:
    """Missing list, update_status, and delete."""

    def add(self, doc: Document) -> None: ...
    def get(self, doc_id: uuid.UUID) -> Document | None: ...
    def get_by_hash(self, content_hash: str) -> Document | None: ...


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


class TestJobRepositoryProtocol:
    def test_valid_stub_satisfies_protocol(self) -> None:
        repo: Any = InMemoryJobRepository()
        assert isinstance(repo, JobRepository)


class TestQueryLogRepositoryProtocol:
    def test_valid_stub_satisfies_protocol(self) -> None:
        repo: Any = InMemoryQueryLogRepository()
        assert isinstance(repo, QueryLogRepository)


# ---------------------------------------------------------------------------
# Tests: basic behavioural correctness of the in-memory stubs
# ---------------------------------------------------------------------------


class TestDocumentRepositoryBehaviour:
    def test_add_and_get(self) -> None:
        repo = InMemoryDocumentRepository()
        doc = _doc()
        repo.add(doc)
        assert repo.get(doc.id) == doc

    def test_get_missing_returns_none(self) -> None:
        repo = InMemoryDocumentRepository()
        assert repo.get(uuid.uuid4()) is None

    def test_get_by_hash(self) -> None:
        repo = InMemoryDocumentRepository()
        doc = _doc()
        repo.add(doc)
        result = repo.get_by_hash(doc.content_hash)
        assert result is not None
        assert result.id == doc.id

    def test_update_status(self) -> None:
        repo = InMemoryDocumentRepository()
        doc = _doc()
        repo.add(doc)
        repo.update_status(doc.id, DocumentStatus.READY)
        updated = repo.get(doc.id)
        assert updated is not None
        assert updated.status == DocumentStatus.READY

    def test_list_pagination(self) -> None:
        repo = InMemoryDocumentRepository()
        for _ in range(5):
            repo.add(_doc())
        page = repo.list(limit=2, offset=0)
        assert len(page) == 2
        page2 = repo.list(limit=2, offset=2)
        assert len(page2) == 2

    def test_list_status_filter(self) -> None:
        repo = InMemoryDocumentRepository()
        doc = _doc()
        repo.add(doc)
        repo.update_status(doc.id, DocumentStatus.READY)
        # Only READY docs
        ready = repo.list(limit=10, offset=0, status=DocumentStatus.READY)
        assert all(d.status == DocumentStatus.READY for d in ready)
        # PENDING filter returns nothing (we updated the only doc)
        pending = repo.list(limit=10, offset=0, status=DocumentStatus.PENDING)
        assert pending == []

    def test_delete(self) -> None:
        repo = InMemoryDocumentRepository()
        doc = _doc()
        repo.add(doc)
        repo.delete(doc.id)
        assert repo.get(doc.id) is None


class TestChunkRepositoryBehaviour:
    def test_add_many_and_list_by_document(self) -> None:
        repo = InMemoryChunkRepository()
        did = uuid.uuid4()
        chunks = [_chunk(did) for _ in range(3)]
        repo.add_many(chunks)
        assert len(repo.list_by_document(did)) == 3

    def test_delete_by_document(self) -> None:
        repo = InMemoryChunkRepository()
        did = uuid.uuid4()
        repo.add_many([_chunk(did), _chunk(did)])
        repo.delete_by_document(did)
        assert repo.list_by_document(did) == []

    def test_count_by_model_version(self) -> None:
        repo = InMemoryChunkRepository()
        did = uuid.uuid4()
        c1 = Chunk(
            id=uuid.uuid4(),
            document_id=did,
            index=0,
            text="a",
            page=None,
            section=None,
            token_count=1,
            embedding_model_version="v1",
        )
        c2 = Chunk(
            id=uuid.uuid4(),
            document_id=did,
            index=1,
            text="b",
            page=None,
            section=None,
            token_count=1,
            embedding_model_version="v2",
        )
        repo.add_many([c1, c2])
        assert repo.count_by_model_version("v1") == 1
        assert repo.count_by_model_version("v2") == 1
        assert repo.count_by_model_version("v3") == 0


class TestJobRepositoryBehaviour:
    def test_create_and_get(self) -> None:
        repo = InMemoryJobRepository()
        job = _job()
        repo.create(job)
        assert repo.get(job.id) == job

    def test_mark_running(self) -> None:
        repo = InMemoryJobRepository()
        job = _job()
        repo.create(job)
        ts = _now()
        repo.mark_running(job.id, ts)
        result = repo.get(job.id)
        assert result is not None
        assert result.status == JobStatus.RUNNING
        assert result.started_at == ts

    def test_mark_succeeded(self) -> None:
        repo = InMemoryJobRepository()
        job = _job()
        repo.create(job)
        ts = _now()
        repo.mark_succeeded(job.id, ts)
        result = repo.get(job.id)
        assert result is not None
        assert result.status == JobStatus.SUCCEEDED
        assert result.finished_at == ts

    def test_mark_failed(self) -> None:
        repo = InMemoryJobRepository()
        job = _job()
        repo.create(job)
        ts = _now()
        repo.mark_failed(job.id, "oops", ts)
        result = repo.get(job.id)
        assert result is not None
        assert result.status == JobStatus.FAILED
        assert result.error == "oops"
        assert result.finished_at == ts

    def test_get_missing_returns_none(self) -> None:
        repo = InMemoryJobRepository()
        assert repo.get(uuid.uuid4()) is None


class TestQueryLogRepositoryBehaviour:
    def test_add_and_list_recent(self) -> None:
        repo = InMemoryQueryLogRepository()
        log = _qlog()
        repo.add(log)
        recent = repo.list_recent(limit=10)
        assert log in recent

    def test_list_recent_respects_limit(self) -> None:
        repo = InMemoryQueryLogRepository()
        for _ in range(5):
            repo.add(_qlog())
        assert len(repo.list_recent(limit=3)) == 3

    def test_add_feedback(self) -> None:
        repo = InMemoryQueryLogRepository()
        log = _qlog()
        repo.add(log)
        repo.add_feedback(log.id, "great answer")
        result = repo.list_recent(limit=1)[0]
        assert result.feedback == "great answer"


# ---------------------------------------------------------------------------
# Extractor Protocol
# ---------------------------------------------------------------------------


class TestExtractorProtocol:
    def test_protocol_conformance(self) -> None:
        from rag.domain.entities import ExtractedDocument
        from rag.domain.interfaces import Extractor

        class DummyExtractor:
            def supports(self, filename: str) -> bool:
                return filename.endswith(".dummy")

            def extract(self, data: bytes, filename: str) -> ExtractedDocument:
                return ExtractedDocument(pages=[], metadata={})

        dummy = DummyExtractor()
        assert isinstance(dummy, Extractor)

    def test_protocol_rejection_missing_methods(self) -> None:
        from rag.domain.interfaces import Extractor

        class IncompleteExtractor:
            def supports(self, filename: str) -> bool:
                return True

        assert not isinstance(IncompleteExtractor(), Extractor)


# ---------------------------------------------------------------------------
# VectorStore Protocol
# ---------------------------------------------------------------------------


class TestVectorStoreProtocol:
    def test_protocol_conformance(self) -> None:
        from rag.domain.entities import SearchHit, VectorRecord
        from rag.domain.interfaces import VectorStore

        class DummyVectorStore:
            def add(self, records: list[VectorRecord]) -> None:
                pass

            def search(
                self,
                vector: list[float],
                k: int,
                filter: dict[str, str | int | float | bool] | None = None,
            ) -> list[SearchHit]:
                return []

            def delete_chunks(self, chunk_ids: list[uuid.UUID]) -> None:
                pass

            def delete_document(self, document_id: uuid.UUID) -> None:
                pass

            def count(self) -> int:
                return 0

            def persist(self) -> None:
                pass

        dummy = DummyVectorStore()
        assert isinstance(dummy, VectorStore)

    def test_protocol_rejection_missing_methods(self) -> None:
        from rag.domain.interfaces import VectorStore

        class IncompleteVectorStore:
            def count(self) -> int:
                return 0

        assert not isinstance(IncompleteVectorStore(), VectorStore)


# ---------------------------------------------------------------------------
# Retrieval Protocols
# ---------------------------------------------------------------------------


class TestRetrieverProtocol:
    def test_protocol_conformance(self) -> None:
        from rag.domain.entities import RetrievalFilter, RetrievalResult
        from rag.domain.interfaces import Retriever

        class DummyRetriever:
            def retrieve(
                self,
                query: str,
                k: int,
                filter: RetrievalFilter | None = None,
            ) -> RetrievalResult:
                return RetrievalResult(chunks=[], trace=[])

        assert isinstance(DummyRetriever(), Retriever)


class TestChunkLookupProtocol:
    def test_protocol_conformance(self) -> None:
        from rag.domain.interfaces import ChunkLookup

        class DummyChunkLookup:
            def get_many(self, chunk_ids: list[uuid.UUID]) -> list[Chunk]:
                return []

        assert isinstance(DummyChunkLookup(), ChunkLookup)


class TestSparseIndexProtocol:
    def test_protocol_conformance(self) -> None:
        from rag.domain.entities import RetrievalFilter
        from rag.domain.interfaces import SparseIndex

        class DummySparseIndex:
            def add(self, chunks: list[Chunk]) -> None:
                pass

            def search(
                self,
                query: str,
                k: int,
                filter: RetrievalFilter | None = None,
            ) -> list[tuple[uuid.UUID, float]]:
                return []

            def delete_document(self, document_id: uuid.UUID) -> None:
                pass

            def count(self) -> int:
                return 0

        assert isinstance(DummySparseIndex(), SparseIndex)


class TestRerankerProtocol:
    def test_protocol_conformance(self) -> None:
        from rag.domain.entities import RetrievedChunk
        from rag.domain.interfaces import Reranker

        class DummyReranker:
            def rerank(
                self,
                query: str,
                chunks: list[RetrievedChunk],
                top_k: int,
            ) -> list[RetrievedChunk]:
                return []

        assert isinstance(DummyReranker(), Reranker)
