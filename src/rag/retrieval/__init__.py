"""Retrieval components: dense, sparse, hybrid, fusion, reranking, and pipeline."""

from __future__ import annotations

from rag.retrieval.bm25 import Bm25Index
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.fakes import FakeReranker, InMemoryChunkLookup
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.pipeline import RetrievalPipeline, build_retriever
from rag.retrieval.reranker import CrossEncoderReranker, resolve_reranker_device
from rag.retrieval.sparse import SparseRetriever

__all__ = [
    "Bm25Index",
    "CrossEncoderReranker",
    "DenseRetriever",
    "FakeReranker",
    "HybridRetriever",
    "InMemoryChunkLookup",
    "RetrievalPipeline",
    "SparseRetriever",
    "build_retriever",
    "reciprocal_rank_fusion",
    "resolve_reranker_device",
]
