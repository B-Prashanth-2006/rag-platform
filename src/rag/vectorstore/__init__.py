"""Vector store package: FAISS and Chroma implementations behind a common protocol."""

from __future__ import annotations

from rag.vectorstore.chroma_store import ChromaVectorStore
from rag.vectorstore.factory import VectorStoreFactory
from rag.vectorstore.faiss_store import FaissVectorStore

__all__ = [
    "ChromaVectorStore",
    "FaissVectorStore",
    "VectorStoreFactory",
]
