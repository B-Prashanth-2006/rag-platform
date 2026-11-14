"""Unit tests for IngestionService with idempotent ingestion."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from rag.domain.entities import (
    Document,
    DocumentStatus,
    IngestionJob,
    JobStatus,
)
from rag.domain.errors import FileTooLargeError
from rag.domain.interfaces import DocumentRepository, JobRepository
from rag.ingestion.registry import ExtractorRegistry
from rag.ingestion.service import IngestionResult, IngestionService


def _now() -> datetime:
    return datetime.now(tz=UTC)


class InMemoryDocumentRepository:
    def __init__(self) -> None:
        self.docs: dict[uuid.UUID, Document] = {}

    def add(self, doc: Document) -> None:
        self.docs[doc.id] = doc

    def get(self, doc_id: uuid.UUID) -> Document | None:
        return self.docs.get(doc_id)

    def get_by_hash(self, content_hash: str) -> Document | None:
        return next((d for d in self.docs.values() if d.content_hash == content_hash), None)

    def list(
        self,
        limit: int,
        offset: int,
        status: DocumentStatus | None = None,
    ) -> list[Document]:
        items = list(self.docs.values())
        if status is not None:
            items = [d for d in items if d.status == status]
        return items[offset : offset + limit]

    def update_status(self, doc_id: uuid.UUID, status: DocumentStatus) -> None:
        if doc_id in self.docs:
            d = self.docs[doc_id]
            self.docs[doc_id] = Document(
                id=d.id,
                filename=d.filename,
                content_hash=d.content_hash,
                status=status,
                metadata=d.metadata,
                created_at=d.created_at,
                updated_at=_now(),
            )

    def delete(self, doc_id: uuid.UUID) -> None:
        self.docs.pop(doc_id, None)


class InMemoryJobRepository:
    def __init__(self) -> None:
        self.jobs: dict[uuid.UUID, IngestionJob] = {}

    def create(self, job: IngestionJob) -> None:
        self.jobs[job.id] = job

    def get(self, job_id: uuid.UUID) -> IngestionJob | None:
        return self.jobs.get(job_id)

    def mark_running(self, job_id: uuid.UUID, started_at: datetime) -> None:
        if job_id in self.jobs:
            j = self.jobs[job_id]
            self.jobs[job_id] = IngestionJob(
                id=j.id,
                document_id=j.document_id,
                status=JobStatus.RUNNING,
                error=None,
                started_at=started_at,
                finished_at=None,
            )

    def mark_succeeded(self, job_id: uuid.UUID, finished_at: datetime) -> None:
        if job_id in self.jobs:
            j = self.jobs[job_id]
            self.jobs[job_id] = IngestionJob(
                id=j.id,
                document_id=j.document_id,
                status=JobStatus.SUCCEEDED,
                error=None,
                started_at=j.started_at,
                finished_at=finished_at,
            )

    def mark_failed(self, job_id: uuid.UUID, error: str, finished_at: datetime) -> None:
        if job_id in self.jobs:
            j = self.jobs[job_id]
            self.jobs[job_id] = IngestionJob(
                id=j.id,
                document_id=j.document_id,
                status=JobStatus.FAILED,
                error=error,
                started_at=j.started_at,
                finished_at=finished_at,
            )


@pytest.fixture
def doc_repo() -> DocumentRepository:
    repo = InMemoryDocumentRepository()
    assert isinstance(repo, DocumentRepository)
    return repo


@pytest.fixture
def job_repo() -> JobRepository:
    repo = InMemoryJobRepository()
    assert isinstance(repo, JobRepository)
    return repo


@pytest.fixture
def registry() -> ExtractorRegistry:
    return ExtractorRegistry()


def test_ingest_new_file_success(
    doc_repo: InMemoryDocumentRepository,
    job_repo: InMemoryJobRepository,
    registry: ExtractorRegistry,
) -> None:
    service = IngestionService(doc_repo=doc_repo, job_repo=job_repo, registry=registry)
    data = b"# Sample Document\n\nThis is a sample document for ingestion."

    result = service.ingest(filename="sample.md", data=data)

    assert isinstance(result, IngestionResult)
    assert result.duplicate is False
    assert result.document.status == DocumentStatus.READY
    assert result.document.filename == "sample.md"

    # Verify job
    assert result.job is not None
    assert result.job.status == JobStatus.SUCCEEDED
    assert result.job.document_id == result.document.id
    assert result.job.started_at is not None
    assert result.job.finished_at is not None
    assert result.job.started_at.tzinfo == UTC
    assert result.job.finished_at.tzinfo == UTC

    # Verify extracted content
    assert result.extracted is not None
    assert len(result.extracted.pages) == 1
    assert "Sample Document" in result.extracted.pages[0].headings

    # Verify persistence in repositories
    assert len(doc_repo.docs) == 1
    assert len(job_repo.jobs) == 1
    assert doc_repo.docs[result.document.id].status == DocumentStatus.READY
    assert job_repo.jobs[result.job.id].status == JobStatus.SUCCEEDED


def test_ingest_duplicate_bytes_idempotent(
    doc_repo: InMemoryDocumentRepository,
    job_repo: InMemoryJobRepository,
    registry: ExtractorRegistry,
) -> None:
    service = IngestionService(doc_repo=doc_repo, job_repo=job_repo, registry=registry)
    data = b"Some repeated plain text content."

    # First ingestion
    first_result = service.ingest(filename="test.txt", data=data)
    assert first_result.duplicate is False
    assert len(doc_repo.docs) == 1
    assert len(job_repo.jobs) == 1

    # Second ingestion with identical bytes
    second_result = service.ingest(filename="test_copy.txt", data=data)
    assert second_result.duplicate is True
    assert second_result.document.id == first_result.document.id
    assert second_result.document.status == DocumentStatus.READY
    assert second_result.job is None
    assert second_result.extracted is None

    # Crucial idempotency check: no new rows created in either repository
    assert len(doc_repo.docs) == 1
    assert len(job_repo.jobs) == 1


def test_ingest_reprocesses_failed_document(
    doc_repo: InMemoryDocumentRepository,
    job_repo: InMemoryJobRepository,
    registry: ExtractorRegistry,
) -> None:
    service = IngestionService(doc_repo=doc_repo, job_repo=job_repo, registry=registry)
    data = b"hello world\n"
    from rag.ingestion.cleaning import compute_content_hash

    content_hash = compute_content_hash(data)

    # Seed a FAILED document in repository
    existing_id = uuid.uuid4()
    doc_repo.add(
        Document(
            id=existing_id,
            filename="test.txt",
            content_hash=content_hash,
            status=DocumentStatus.FAILED,
            metadata={},
            created_at=_now(),
            updated_at=_now(),
        )
    )

    result = service.ingest(filename="test.txt", data=data)

    assert result.duplicate is False
    # Reuses existing document ID
    assert result.document.id == existing_id
    assert result.document.status == DocumentStatus.READY
    assert doc_repo.docs[existing_id].status == DocumentStatus.READY

    # Created a new succeeded job
    assert result.job is not None
    assert result.job.document_id == existing_id
    assert result.job.status == JobStatus.SUCCEEDED
    assert len(doc_repo.docs) == 1
    assert len(job_repo.jobs) == 1


def test_ingest_corrupt_file_marks_failed_without_raising(
    doc_repo: InMemoryDocumentRepository,
    job_repo: InMemoryJobRepository,
    registry: ExtractorRegistry,
) -> None:
    service = IngestionService(doc_repo=doc_repo, job_repo=job_repo, registry=registry)
    corrupt_data = b"%PDF-1.4 corrupt invalid garbage bytes"

    result = service.ingest(filename="corrupt.pdf", data=corrupt_data)

    assert result.duplicate is False
    assert result.document.status == DocumentStatus.FAILED
    assert result.job is not None
    assert result.job.status == JobStatus.FAILED
    assert result.job.error is not None
    assert len(result.job.error) > 0
    assert result.extracted is None

    # Stored state reflects failure
    assert doc_repo.docs[result.document.id].status == DocumentStatus.FAILED
    assert job_repo.jobs[result.job.id].status == JobStatus.FAILED
    assert job_repo.jobs[result.job.id].error == result.job.error


def test_ingest_file_too_large(
    doc_repo: InMemoryDocumentRepository,
    job_repo: InMemoryJobRepository,
    registry: ExtractorRegistry,
) -> None:
    # Set max_upload_mb to 1 MB for testing
    service = IngestionService(
        doc_repo=doc_repo, job_repo=job_repo, registry=registry, max_upload_mb=1
    )
    oversized_data = b"x" * (1024 * 1024 + 10)  # slightly over 1 MB

    with pytest.raises(FileTooLargeError) as exc_info:
        service.ingest(filename="large.txt", data=oversized_data)

    assert exc_info.value.size_bytes == len(oversized_data)
    assert exc_info.value.max_bytes == 1024 * 1024

    # Repositories were never touched
    assert len(doc_repo.docs) == 0
    assert len(job_repo.jobs) == 0
