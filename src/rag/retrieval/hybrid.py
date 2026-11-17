"""Hybrid retriever combining dense semantic search and sparse BM25 keyword search."""

from __future__ import annotations

import time

from rag.domain.entities import (
    RetrievalFilter,
    RetrievalResult,
    StageTrace,
)
from rag.domain.interfaces import Retriever
from rag.retrieval.fusion import reciprocal_rank_fusion


class HybridRetriever:
    """Hybrid retriever combining dense and sparse retrievers with Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: Retriever,
        sparse_retriever: Retriever,
        candidate_k: int = 50,
        fusion_k: int = 10,
        rrf_k: int = 60,
        weights: list[float] | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.candidate_k = candidate_k
        self.fusion_k = fusion_k
        self.rrf_k = rrf_k
        self.weights = weights

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter: RetrievalFilter | None = None,
    ) -> RetrievalResult:
        """Retrieve candidates from dense and sparse retrievers and fuse with RRF."""
        effective_k = min(k if k is not None else self.fusion_k, self.fusion_k)

        if effective_k <= 0 or not query.strip():
            # Run retrievers to preserve standard diagnostic traces
            dense_res = self.dense_retriever.retrieve(query=query, k=0, filter=filter)
            sparse_res = self.sparse_retriever.retrieve(query=query, k=0, filter=filter)
            fusion_trace = StageTrace(stage="fused", latency_ms=0.0, results=[])
            return RetrievalResult(
                chunks=[],
                trace=list(dense_res.trace) + list(sparse_res.trace) + [fusion_trace],
            )

        # Retrieve candidate pools from both branches
        dense_res = self.dense_retriever.retrieve(
            query=query,
            k=self.candidate_k,
            filter=filter,
        )
        sparse_res = self.sparse_retriever.retrieve(
            query=query,
            k=self.candidate_k,
            filter=filter,
        )

        t_fusion_start = time.perf_counter()
        fused_chunks = reciprocal_rank_fusion(
            ranked_lists=[dense_res.chunks, sparse_res.chunks],
            k=self.rrf_k,
            weights=self.weights,
        )
        fusion_latency_ms = (time.perf_counter() - t_fusion_start) * 1000.0

        selected_chunks = fused_chunks[:effective_k]
        fusion_results = [(c.chunk_id, c.score) for c in selected_chunks]
        fusion_trace = StageTrace(
            stage="fused",
            latency_ms=fusion_latency_ms,
            results=fusion_results,
        )

        combined_trace = list(dense_res.trace) + list(sparse_res.trace) + [fusion_trace]
        return RetrievalResult(
            chunks=selected_chunks,
            trace=combined_trace,
        )
