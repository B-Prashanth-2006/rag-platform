"""IngestionService: orchestrates idempotent document extraction and cleaning."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from rag.config import get_settings
from rag.domain.entities import (
    Document,
    DocumentStatus,
    ExtractedDocument,
    IngestionJob,
    JobStatus,
)
from rag.domain.errors import FileTooLargeError
from rag.domain.interfaces import DocumentRepository, JobRepository
from rag.ingestion.cleaning import clean_document, compute_content_hash
from rag.ingestion.registry import ExtractorRegistry


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""

    document: Document
    job: IngestionJob | None
    duplicate: bool
    extracted: ExtractedDocument | None


class IngestionService:
    """Coordinates document ingestion: duplicate detection, extraction, and cleaning."""

    def __init__(
        self,
        doc_repo: DocumentRepository,
        job_repo: JobRepository,
        registry: ExtractorRegistry,
        max_upload_mb: int | None = None,
    ) -> None:
        self._doc_repo = doc_repo
        self._job_repo = job_repo
        self._registry = registry
        limit_mb = max_upload_mb if max_upload_mb is not None else get_settings().max_upload_mb
        self._max_upload_bytes = limit_mb * 1024 * 1024

    def ingest(self, filename: str, data: bytes) -> IngestionResult:
        """Ingest raw file bytes into the system idempotently.

        If a document with the same content hash exists and is not FAILED, returns
        it as duplicate without modifying database rows.
        If an existing document is FAILED, it is reprocessed with a new job.
        Otherwise creates a new document and ingestion job, extracts, cleans, and marks READY.
        On extraction error, marks the document and job FAILED without raising.
        """
        data_len = len(data)
        if data_len > self._max_upload_bytes:
            raise FileTooLargeError(size_bytes=data_len, max_bytes=self._max_upload_bytes)

        content_hash = compute_content_hash(data)
        existing_doc = self._doc_repo.get_by_hash(content_hash)

        # 1. Duplicate check: existing document that is not FAILED
        if existing_doc is not None and existing_doc.status != DocumentStatus.FAILED:
            return IngestionResult(
                document=existing_doc,
                job=None,
                duplicate=True,
                extracted=None,
            )

        now = datetime.now(tz=UTC)

        # 2. Re-process FAILED document or create a new one
        if existing_doc is not None and existing_doc.status == DocumentStatus.FAILED:
            doc_id = existing_doc.id
            self._doc_repo.update_status(doc_id, DocumentStatus.PROCESSING)
            doc = Document(
                id=doc_id,
                filename=filename,
                content_hash=content_hash,
                status=DocumentStatus.PROCESSING,
                metadata=dict(existing_doc.metadata),
                created_at=existing_doc.created_at,
                updated_at=now,
            )
        else:
            doc_id = uuid.uuid4()
            doc = Document(
                id=doc_id,
                filename=filename,
                content_hash=content_hash,
                status=DocumentStatus.PROCESSING,
                metadata={},
                created_at=now,
                updated_at=now,
            )
            self._doc_repo.add(doc)

        job = IngestionJob(
            id=uuid.uuid4(),
            document_id=doc_id,
            status=JobStatus.QUEUED,
            error=None,
            started_at=None,
            finished_at=None,
        )
        self._job_repo.create(job)

        # 3. Mark running
        started_at = datetime.now(tz=UTC)
        self._job_repo.mark_running(job.id, started_at=started_at)
        job.status = JobStatus.RUNNING
        job.started_at = started_at

        # 4. Extract and clean
        try:
            raw_doc = self._registry.extract(data, filename)
            cleaned_doc = clean_document(raw_doc)

            finished_at = datetime.now(tz=UTC)
            self._job_repo.mark_succeeded(job.id, finished_at=finished_at)
            job.status = JobStatus.SUCCEEDED
            job.finished_at = finished_at

            self._doc_repo.update_status(doc.id, DocumentStatus.READY)
            doc.status = DocumentStatus.READY
            doc.updated_at = finished_at

            return IngestionResult(
                document=doc,
                job=job,
                duplicate=False,
                extracted=cleaned_doc,
            )
        except Exception as exc:
            finished_at = datetime.now(tz=UTC)
            error_message = str(exc)
            self._job_repo.mark_failed(job.id, error=error_message, finished_at=finished_at)
            job.status = JobStatus.FAILED
            job.error = error_message
            job.finished_at = finished_at

            self._doc_repo.update_status(doc.id, DocumentStatus.FAILED)
            doc.status = DocumentStatus.FAILED
            doc.updated_at = finished_at

            return IngestionResult(
                document=doc,
                job=job,
                duplicate=False,
                extracted=None,
            )
