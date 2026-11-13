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
