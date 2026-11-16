"""Sparse keyword retriever combining SparseIndex and ChunkLookup."""

from __future__ import annotations

import time

from rag.domain.entities import (
    RetrievalFilter,
    RetrievalResult,
    RetrievedChunk,
    StageTrace,
)
from rag.domain.interfaces import ChunkLookup, SparseIndex


class SparseRetriever:
    """Keyword retriever implementing the Retriever protocol via SparseIndex."""

    def __init__(self, index: SparseIndex, chunk_lookup: ChunkLookup) -> None:
        self.index = index
        self.chunk_lookup = chunk_lookup

    def retrieve(
        self,
        query: str,
        k: int,
        filter: RetrievalFilter | None = None,
    ) -> RetrievalResult:
        """Retrieve top-k chunks using sparse BM25 keyword search."""
        t_start = time.perf_counter()

        if k <= 0 or not query.strip():
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return RetrievalResult(
                chunks=[],
                trace=[StageTrace(stage="sparse", latency_ms=latency_ms, results=[])],
            )

        hits = self.index.search(query=query, k=k, filter=filter)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if not hits:
            return RetrievalResult(
                chunks=[],
                trace=[StageTrace(stage="sparse", latency_ms=latency_ms, results=[])],
            )

        score_map = dict(hits)
        hydrated_chunks = self.chunk_lookup.get_many([cid for cid, _ in hits])

        retrieved: list[RetrievedChunk] = []
        for c in hydrated_chunks:
            retrieved.append(
                RetrievedChunk(
                    chunk_id=c.id,
                    document_id=c.document_id,
                    text=c.text,
                    page=c.page,
                    section=c.section,
                    score=score_map.get(c.id, 0.0),
                    source="sparse",
                )
            )

        trace = StageTrace(stage="sparse", latency_ms=latency_ms, results=hits)
        return RetrievalResult(chunks=retrieved, trace=[trace])
