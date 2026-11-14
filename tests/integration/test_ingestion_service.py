"""Integration tests for IngestionService with real PostgreSQL repositories."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag.db.models import DocumentOrm, IngestionJobOrm
from rag.db.repositories.documents import SqlDocumentRepository
from rag.db.repositories.jobs import SqlJobRepository
from rag.domain.entities import Document, DocumentStatus, JobStatus
from rag.ingestion.cleaning import compute_content_hash
from rag.ingestion.registry import ExtractorRegistry
from rag.ingestion.service import IngestionResult, IngestionService


def test_end_to_end_ingestion_and_idempotency(session: Session) -> None:
    doc_repo = SqlDocumentRepository(session)
    job_repo = SqlJobRepository(session)
    registry = ExtractorRegistry()
    service = IngestionService(doc_repo=doc_repo, job_repo=job_repo, registry=registry)

    data = b"# Quarterly Overview\n\nOperational metrics for Q1 2026."

    # 1. New file ingested to READY with a succeeded job
    res1 = service.ingest(filename="q1.md", data=data)
    session.commit()

    assert isinstance(res1, IngestionResult)
    assert res1.duplicate is False
    assert res1.document.status == DocumentStatus.READY
    assert res1.job is not None
    assert res1.job.status == JobStatus.SUCCEEDED
    assert res1.extracted is not None
    assert len(res1.extracted.pages) == 1

    # Verify rows in PostgreSQL
    doc_count = session.scalar(select(func.count()).select_from(DocumentOrm))
    job_count = session.scalar(select(func.count()).select_from(IngestionJobOrm))
    assert doc_count == 1
    assert job_count == 1

    # 2. Same bytes uploaded again returns duplicate=True and creates no rows
    res2 = service.ingest(filename="q1_copy.md", data=data)
    session.commit()

    assert res2.duplicate is True
    assert res2.document.id == res1.document.id
    assert res2.job is None
    assert res2.extracted is None

    # Verify DB row counts did not change
    doc_count_after = session.scalar(select(func.count()).select_from(DocumentOrm))
    job_count_after = session.scalar(select(func.count()).select_from(IngestionJobOrm))
    assert doc_count_after == 1
    assert job_count_after == 1


def test_corrupt_file_and_reprocess_failed_document(session: Session) -> None:
    doc_repo = SqlDocumentRepository(session)
    job_repo = SqlJobRepository(session)
    registry = ExtractorRegistry()
    service = IngestionService(doc_repo=doc_repo, job_repo=job_repo, registry=registry)

    corrupt_data = b"%PDF-1.4 corrupt invalid bytes"

    # 1. Corrupt file ends FAILED with error recorded and no exception raised
    res_failed = service.ingest(filename="broken.pdf", data=corrupt_data)
    session.commit()

    assert res_failed.duplicate is False
    assert res_failed.document.status == DocumentStatus.FAILED
    assert res_failed.job is not None
    assert res_failed.job.status == JobStatus.FAILED
    assert res_failed.job.error is not None
    assert "Failed to parse PDF" in res_failed.job.error or "corrupt" in res_failed.job.error

    failed_doc_id = res_failed.document.id

    # 2. A FAILED document can be reprocessed
    # Simulate valid bytes matching the same content hash or updating the file
    # We test that a failed doc gets reprocessed when ingested again:
    res_retry = service.ingest(filename="broken.pdf", data=corrupt_data)
    session.commit()

    # Document ID is reused
    assert res_retry.document.id == failed_doc_id
    assert res_retry.duplicate is False
    # Still corrupt, so it fails again with a new job
    assert res_retry.job is not None
    assert res_retry.job.status == JobStatus.FAILED

    # Total jobs for this document is 2
    jobs = session.scalars(
        select(IngestionJobOrm).where(IngestionJobOrm.document_id == failed_doc_id)
    ).all()
    assert len(jobs) == 2


def test_reprocess_failed_document_ends_ready(session: Session) -> None:
    doc_repo = SqlDocumentRepository(session)
    job_repo = SqlJobRepository(session)
    registry = ExtractorRegistry()
    service = IngestionService(doc_repo=doc_repo, job_repo=job_repo, registry=registry)

    valid_data = b"# Reprocess Me\n\nRecovered document content."
    content_hash = compute_content_hash(valid_data)

    # Seed an existing FAILED document in the database
    failed_doc = Document(
        id=uuid.uuid4(),
        filename="recover.md",
        content_hash=content_hash,
        status=DocumentStatus.FAILED,
        metadata={},
        created_at=datetime.now(tz=UTC),
        updated_at=datetime.now(tz=UTC),
    )
    doc_repo.add(failed_doc)
    session.commit()

    # Re-ingest the same bytes
    result = service.ingest(filename="recover.md", data=valid_data)
    session.commit()

    # Must reuse document ID, not be duplicate, end READY with SUCCEEDED job
    assert result.duplicate is False
    assert result.document.id == failed_doc.id
    assert result.document.status == DocumentStatus.READY
    assert result.job is not None
    assert result.job.status == JobStatus.SUCCEEDED

    # Verify DB state
    db_doc = doc_repo.get(failed_doc.id)
    assert db_doc is not None
    assert db_doc.status == DocumentStatus.READY


def test_service_never_commits(session: Session) -> None:
    doc_repo = SqlDocumentRepository(session)
    job_repo = SqlJobRepository(session)
    registry = ExtractorRegistry()
    service = IngestionService(doc_repo=doc_repo, job_repo=job_repo, registry=registry)

    data = b"Uncommitted test data content."
    res = service.ingest(filename="uncommitted.txt", data=data)
    assert res.duplicate is False

    # Rollback without committing
    session.rollback()

    # Verify nothing was persisted because the service never committed
    row = session.get(DocumentOrm, res.document.id)
    assert row is None
