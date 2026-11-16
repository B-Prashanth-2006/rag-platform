"""In-memory fakes for retrieval testing."""

from __future__ import annotations

import uuid

from rag.domain.entities import Chunk


class InMemoryChunkLookup:
    """In-memory ChunkLookup for unit tests."""

    def __init__(self, initial_chunks: list[Chunk] | None = None) -> None:
        self._store: dict[uuid.UUID, Chunk] = {}
        if initial_chunks:
            for c in initial_chunks:
                self._store[c.id] = c

    def add(self, chunk: Chunk) -> None:
        """Add a chunk to the in-memory store."""
        self._store[chunk.id] = chunk

    def add_many(self, chunks: list[Chunk]) -> None:
        """Add multiple chunks to the in-memory store."""
        for c in chunks:
            self._store[c.id] = c

    def get_many(self, chunk_ids: list[uuid.UUID]) -> list[Chunk]:
        """Return chunks in requested order, skipping missing IDs."""
        return [self._store[cid] for cid in chunk_ids if cid in self._store]
