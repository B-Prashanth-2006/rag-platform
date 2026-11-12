"""Domain repository Protocols for the RAG platform.

All Protocols are @runtime_checkable so structural subtyping can be verified
in tests (isinstance checks). No implementation details live here — only
method signatures.

Rule: this module imports nothing from other ``rag`` packages.
"""

from __future__ import annotations

import uuid
from typing import Protocol, runtime_checkable

from rag.domain.entities import Chunk, Document, DocumentStatus, IngestionJob, QueryLog


@runtime_checkable
class DocumentRepository(Protocol):
    """Persistence contract for :class:`~rag.domain.entities.Document` aggregates."""

    async def add(self, doc: Document) -> None:
        """Persist a new document record."""
        ...

    async def get(self, doc_id: uuid.UUID) -> Document | None:
        """Return the document with *doc_id*, or ``None`` if absent."""
        ...

    async def get_by_hash(self, content_hash: str) -> Document | None:
        """Return the document whose ``content_hash`` matches, or ``None``."""
        ...

    async def update_status(self, doc_id: uuid.UUID, status: DocumentStatus) -> None:
        """Overwrite the status field of an existing document."""
        ...

    async def list_all(self, limit: int, offset: int) -> list[Document]:
        """Return a page of documents ordered by insertion time."""
        ...


@runtime_checkable
class ChunkRepository(Protocol):
    """Persistence contract for :class:`~rag.domain.entities.Chunk` records."""

    async def add_many(self, chunks: list[Chunk]) -> None:
        """Bulk-insert a list of chunks (all belonging to the same document)."""
        ...

    async def get_by_document(self, document_id: uuid.UUID) -> list[Chunk]:
        """Return all chunks for *document_id*, ordered by ``index``."""
        ...

    async def delete_by_document(self, document_id: uuid.UUID) -> None:
        """Remove every chunk associated with *document_id*."""
        ...


@runtime_checkable
class IngestionJobRepository(Protocol):
    """Persistence contract for :class:`~rag.domain.entities.IngestionJob` records."""

    async def add(self, job: IngestionJob) -> None:
        """Persist a new ingestion job."""
        ...

    async def get(self, job_id: uuid.UUID) -> IngestionJob | None:
        """Return the job with *job_id*, or ``None`` if absent."""
        ...

    async def update(self, job: IngestionJob) -> None:
        """Overwrite the entire job record (status, error, timestamps)."""
        ...


@runtime_checkable
class QueryLogRepository(Protocol):
    """Persistence contract for :class:`~rag.domain.entities.QueryLog` records."""

    async def add(self, log: QueryLog) -> None:
        """Persist a new query log entry."""
        ...

    async def get(self, log_id: uuid.UUID) -> QueryLog | None:
        """Return the log entry with *log_id*, or ``None`` if absent."""
        ...
