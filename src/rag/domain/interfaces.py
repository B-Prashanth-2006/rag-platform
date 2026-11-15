"""Domain repository Protocols for the RAG platform.

All Protocols are @runtime_checkable so structural subtyping can be verified
in tests (isinstance checks). No implementation details live here — only
method signatures.

Rule: this module imports nothing from other ``rag`` packages.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Protocol, runtime_checkable

from rag.domain.entities import (
    Chunk,
    ChunkDraft,
    Document,
    DocumentStatus,
    ExtractedDocument,
    IngestionJob,
    JobStatus,
    QueryLog,
)


@runtime_checkable
class DocumentRepository(Protocol):
    """Persistence contract for :class:`~rag.domain.entities.Document` aggregates."""

    def add(self, doc: Document) -> None:
        """Persist a new document record."""
        ...

    def get(self, doc_id: uuid.UUID) -> Document | None:
        """Return the document with *doc_id*, or ``None`` if absent."""
        ...

    def get_by_hash(self, content_hash: str) -> Document | None:
        """Return the document whose ``content_hash`` matches, or ``None``."""
        ...

    def list(
        self,
        limit: int,
        offset: int,
        status: DocumentStatus | None = None,
    ) -> list[Document]:
        """Return a page of documents ordered by insertion time, optionally filtered by status."""
        ...

    def update_status(self, doc_id: uuid.UUID, status: DocumentStatus) -> None:
        """Overwrite the status field of an existing document."""
        ...

    def delete(self, doc_id: uuid.UUID) -> None:
        """Remove the document with *doc_id* from the store."""
        ...


@runtime_checkable
class ChunkRepository(Protocol):
    """Persistence contract for :class:`~rag.domain.entities.Chunk` records."""

    def add_many(self, chunks: list[Chunk]) -> None:
        """Bulk-insert a list of chunks (all belonging to the same document)."""
        ...

    def list_by_document(self, document_id: uuid.UUID) -> list[Chunk]:
        """Return all chunks for *document_id*, ordered by ``index``."""
        ...

    def delete_by_document(self, document_id: uuid.UUID) -> None:
        """Remove every chunk associated with *document_id*."""
        ...

    def count_by_model_version(self, version: str) -> int:
        """Return the number of chunks tagged with *embedding_model_version*."""
        ...


@runtime_checkable
class JobRepository(Protocol):
    """Persistence contract for :class:`~rag.domain.entities.IngestionJob` records."""

    def create(self, job: IngestionJob) -> None:
        """Persist a new ingestion job."""
        ...

    def get(self, job_id: uuid.UUID) -> IngestionJob | None:
        """Return the job with *job_id*, or ``None`` if absent."""
        ...

    def mark_running(self, job_id: uuid.UUID, started_at: datetime) -> None:
        """Transition the job to RUNNING and record *started_at*."""
        ...

    def mark_succeeded(self, job_id: uuid.UUID, finished_at: datetime) -> None:
        """Transition the job to SUCCEEDED and record *finished_at*."""
        ...

    def mark_failed(self, job_id: uuid.UUID, error: str, finished_at: datetime) -> None:
        """Transition the job to FAILED, record *error* and *finished_at*."""
        ...


@runtime_checkable
class QueryLogRepository(Protocol):
    """Persistence contract for :class:`~rag.domain.entities.QueryLog` records."""

    def add(self, log: QueryLog) -> None:
        """Persist a new query log entry."""
        ...

    def add_feedback(self, log_id: uuid.UUID, feedback: str) -> None:
        """Attach user feedback to an existing log entry."""
        ...

    def list_recent(self, limit: int) -> list[QueryLog]:
        """Return the *limit* most recent query log entries, newest first."""
        ...


@runtime_checkable
class Extractor(Protocol):
    """Extraction contract for converting raw file bytes into structured documents."""

    def supports(self, filename: str) -> bool:
        """Return True if this extractor can handle the given filename/extension."""
        ...

    def extract(self, data: bytes, filename: str) -> ExtractedDocument:
        """Extract pages, headings, tables and metadata from raw file bytes."""
        ...


@runtime_checkable
class TokenCounter(Protocol):
    """Contract for counting, encoding, and decoding tokens."""

    def count(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        ...

    def encode(self, text: str) -> list[int]:
        """Convert *text* into a sequence of token IDs."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Reconstruct text from a sequence of token IDs."""
        ...


@runtime_checkable
class Chunker(Protocol):
    """Contract for splitting an extracted document into chunk drafts."""

    def chunk(self, doc: ExtractedDocument) -> list[ChunkDraft]:
        """Split *doc* into contiguous chunk drafts with metadata."""
        ...


# Re-export JobStatus so callers can import it alongside JobRepository.
__all__ = [
    "ChunkRepository",
    "Chunker",
    "DocumentRepository",
    "Extractor",
    "JobRepository",
    "JobStatus",
    "QueryLogRepository",
    "TokenCounter",
]
