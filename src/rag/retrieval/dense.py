"""Dense vector retriever combining Embedder, VectorStore, and ChunkLookup."""

from __future__ import annotations

import time

from rag.domain.entities import (
    RetrievalFilter,
    RetrievalResult,
    RetrievedChunk,
    StageTrace,
)
from rag.domain.interfaces import ChunkLookup, Embedder, VectorStore


class DenseRetriever:
    """Dense semantic retriever implementing the Retriever protocol via VectorStore."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        chunk_lookup: ChunkLookup,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunk_lookup = chunk_lookup

    def retrieve(
        self,
        query: str,
        k: int,
        filter: RetrievalFilter | None = None,
    ) -> RetrievalResult:
        """Retrieve top-k chunks using dense embedding vector similarity."""
        t_start = time.perf_counter()

        if k <= 0 or not query.strip():
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return RetrievalResult(
                chunks=[],
                trace=[StageTrace(stage="dense", latency_ms=latency_ms, results=[])],
            )

        query_vector = self.embedder.embed_query(query)

        # Prepare vector store filter if single document_id is specified
        vs_filter: dict[str, str | int | float | bool] | None = None
        has_multi_doc_filter = False
        allowed_docs = None

        if filter and filter.document_ids:
            allowed_docs = set(filter.document_ids)
            if len(filter.document_ids) == 1:
                vs_filter = {"document_id": str(filter.document_ids[0])}
            else:
                has_multi_doc_filter = True

        fetch_k = max(k * 4, 50) if has_multi_doc_filter else k
        hits = self.vector_store.search(vector=query_vector, k=fetch_k, filter=vs_filter)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if not hits:
            return RetrievalResult(
                chunks=[],
                trace=[StageTrace(stage="dense", latency_ms=latency_ms, results=[])],
            )

        score_map = {h.chunk_id: h.score for h in hits}
        hydrated_chunks = self.chunk_lookup.get_many([h.chunk_id for h in hits])

        retrieved: list[RetrievedChunk] = []
        for c in hydrated_chunks:
            if allowed_docs is not None and c.document_id not in allowed_docs:
                continue

            retrieved.append(
                RetrievedChunk(
                    chunk_id=c.id,
                    document_id=c.document_id,
                    text=c.text,
                    page=c.page,
                    section=c.section,
                    score=score_map.get(c.id, 0.0),
                    source="dense",
                )
            )
            if len(retrieved) == k:
                break

        results = [(rc.chunk_id, rc.score) for rc in retrieved]
        trace = StageTrace(stage="dense", latency_ms=latency_ms, results=results)
        return RetrievalResult(chunks=retrieved, trace=[trace])
