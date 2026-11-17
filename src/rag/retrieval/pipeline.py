"""Retrieval pipeline and build_retriever factory function."""

from __future__ import annotations

import time

from rag.config import Settings, get_settings
from rag.domain.entities import (
    RetrievalFilter,
    RetrievalResult,
    StageTrace,
)
from rag.domain.interfaces import (
    ChunkLookup,
    Embedder,
    Reranker,
    Retriever,
    SparseIndex,
    VectorStore,
)
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranker import CrossEncoderReranker
from rag.retrieval.sparse import SparseRetriever


class RetrievalPipeline:
    """Composable multi-stage retrieval pipeline with diagnostic tracing."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker | None = None,
        use_reranker: bool = False,
        rerank_candidates: int = 20,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.use_reranker = use_reranker
        self.rerank_candidates = rerank_candidates

    def retrieve(
        self,
        query: str,
        k: int = 10,
        filter: RetrievalFilter | None = None,
    ) -> RetrievalResult:
        """Execute the retrieval pipeline: base retrieval followed by optional reranking."""
        if k <= 0 or not query.strip():
            base_result = self.retriever.retrieve(query=query, k=0, filter=filter)
            trace = list(base_result.trace)
            if self.use_reranker:
                trace.append(StageTrace(stage="reranked", latency_ms=0.0, results=[]))
            return RetrievalResult(chunks=[], trace=trace)

        if self.use_reranker and self.reranker is not None:
            fetch_k = max(k, self.rerank_candidates)
            base_result = self.retriever.retrieve(query=query, k=fetch_k, filter=filter)
            if not base_result.chunks:
                trace = list(base_result.trace)
                trace.append(StageTrace(stage="reranked", latency_ms=0.0, results=[]))
                return RetrievalResult(chunks=[], trace=trace)

            t0 = time.perf_counter()
            reranked_chunks = self.reranker.rerank(
                query=query,
                chunks=base_result.chunks,
                top_k=k,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            rerank_trace = StageTrace(
                stage="reranked",
                latency_ms=latency_ms,
                results=[(c.chunk_id, c.score) for c in reranked_chunks],
            )
            return RetrievalResult(
                chunks=reranked_chunks,
                trace=list(base_result.trace) + [rerank_trace],
            )

        base_result = self.retriever.retrieve(query=query, k=k, filter=filter)
        return RetrievalResult(
            chunks=base_result.chunks[:k],
            trace=base_result.trace,
        )


def build_retriever(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
    sparse_index: SparseIndex | None = None,
    chunk_lookup: ChunkLookup | None = None,
    reranker: Reranker | None = None,
) -> RetrievalPipeline:
    """Factory function constructing a configured RetrievalPipeline."""
    cfg = settings or get_settings()
    mode = cfg.retrieval_mode.lower()

    base_retriever: Retriever
    if mode == "dense":
        if embedder is None or vector_store is None or chunk_lookup is None:
            raise ValueError("Dense retrieval requires embedder, vector_store, and chunk_lookup")
        base_retriever = DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            chunk_lookup=chunk_lookup,
        )
    elif mode == "sparse":
        if sparse_index is None or chunk_lookup is None:
            raise ValueError("Sparse retrieval requires sparse_index and chunk_lookup")
        base_retriever = SparseRetriever(
            index=sparse_index,
            chunk_lookup=chunk_lookup,
        )
    elif mode == "hybrid":
        if embedder is None or vector_store is None or sparse_index is None or chunk_lookup is None:
            raise ValueError(
                "Hybrid retrieval requires embedder, vector_store, sparse_index, and chunk_lookup"
            )
        dense_ret = DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            chunk_lookup=chunk_lookup,
        )
        sparse_ret = SparseRetriever(
            index=sparse_index,
            chunk_lookup=chunk_lookup,
        )
        base_retriever = HybridRetriever(
            dense_retriever=dense_ret,
            sparse_retriever=sparse_ret,
            candidate_k=cfg.candidate_k,
            fusion_k=cfg.fusion_k,
            rrf_k=cfg.rrf_k,
        )
    else:
        raise ValueError(
            f"Unknown retrieval mode '{cfg.retrieval_mode}'. "
            "Must be 'dense', 'sparse', or 'hybrid'."
        )

    resolved_reranker: Reranker | None = None
    if cfg.use_reranker:
        if reranker is not None:
            resolved_reranker = reranker
        else:
            resolved_reranker = CrossEncoderReranker(
                model_name=cfg.reranker_model,
                device=cfg.reranker_device,
                batch_size=cfg.reranker_batch_size,
                rerank_candidates=cfg.rerank_candidates,
            )

    return RetrievalPipeline(
        retriever=base_retriever,
        reranker=resolved_reranker,
        use_reranker=cfg.use_reranker,
        rerank_candidates=cfg.rerank_candidates,
    )
