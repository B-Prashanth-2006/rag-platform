"""In-memory fakes for retrieval testing."""

from __future__ import annotations

import uuid
from collections.abc import Callable

from rag.domain.entities import Chunk, RetrievedChunk


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


class FakeReranker:
    """In-memory Reranker for unit tests."""

    def __init__(
        self,
        rerank_candidates: int = 20,
        score_fn: Callable[[str, RetrievedChunk], float] | None = None,
    ) -> None:
        self.rerank_candidates = rerank_candidates
        self.score_fn = score_fn

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank chunks in memory according to score_fn or reciprocal index."""
        if not chunks or top_k <= 0 or not query.strip():
            return []

        candidates = chunks[: self.rerank_candidates]
        scored: list[tuple[RetrievedChunk, float]] = []
        for i, c in enumerate(candidates):
            score = self.score_fn(query, c) if self.score_fn is not None else 1.0 / (i + 1)
            scored.append((c, score))

        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
        selected = scored[:top_k]

        return [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                page=c.page,
                section=c.section,
                score=score,
                source="reranked",
            )
            for c, score in selected
        ]
