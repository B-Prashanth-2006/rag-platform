"""Retrieval components: dense, sparse, fusion, and in-memory fakes."""

from __future__ import annotations

from rag.retrieval.bm25 import Bm25Index
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.fakes import InMemoryChunkLookup
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.sparse import SparseRetriever

__all__ = [
    "Bm25Index",
    "DenseRetriever",
    "InMemoryChunkLookup",
    "SparseRetriever",
    "reciprocal_rank_fusion",
]
