"""Domain error types for the RAG platform.

These are plain Python exceptions — no SQLAlchemy imports, no external dependencies.
"""

from __future__ import annotations

import uuid


class DuplicateContentHashError(Exception):
    """Raised when a document with the same content_hash already exists."""

    def __init__(self, content_hash: str) -> None:
        self.content_hash = content_hash
        super().__init__(f"A document with content_hash {content_hash!r} already exists.")


class DocumentNotFoundError(Exception):
    """Raised when a document with the given id cannot be found."""

    def __init__(self, doc_id: uuid.UUID) -> None:
        self.doc_id = doc_id
        super().__init__(f"No document found with id {doc_id}.")


class DuplicateChunkIndexError(Exception):
    """Raised when a chunk with the same (document_id, index) already exists."""

    def __init__(self, document_id: uuid.UUID, index: int) -> None:
        self.document_id = document_id
        self.index = index
        super().__init__(f"Chunk index {index} already exists for document {document_id}.")


class JobNotFoundError(Exception):
    """Raised when an ingestion job with the given id cannot be found."""

    def __init__(self, job_id: uuid.UUID) -> None:
        self.job_id = job_id
        super().__init__(f"No ingestion job found with id {job_id}.")


class QueryLogNotFoundError(Exception):
    """Raised when a query log entry with the given id cannot be found."""

    def __init__(self, log_id: uuid.UUID) -> None:
        self.log_id = log_id
        super().__init__(f"No query log found with id {log_id}.")


class UnsupportedFileTypeError(Exception):
    """Raised when an uploaded file's type or extension is not supported."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(f"Unsupported file type for {filename!r}.")


class FileTooLargeError(Exception):
    """Raised when an uploaded file exceeds the configured maximum upload size."""

    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        super().__init__(
            f"File size {size_bytes} bytes exceeds maximum allowed size of {max_bytes} bytes."
        )


class ExtractionError(Exception):
    """Raised when an error occurs during extraction (corrupt, empty, or unreadable file)."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
