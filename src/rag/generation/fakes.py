"""In-memory substitutes for generation dependencies."""

from __future__ import annotations

import uuid


class InMemoryDocumentLookup:
    """Resolve document IDs to names without a database."""

    def __init__(self, names: dict[uuid.UUID, str] | None = None) -> None:
        self._names = dict(names or {})

    def add(self, document_id: uuid.UUID, name: str) -> None:
        """Store or replace a document name."""
        self._names[document_id] = name

    def get_names(self, ids: list[uuid.UUID]) -> dict[uuid.UUID, str]:
        """Return found names in the order requested, skipping unknown IDs."""
        return {
            document_id: self._names[document_id]
            for document_id in ids
            if document_id in self._names
        }
