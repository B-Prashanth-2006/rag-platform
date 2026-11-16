"""Factory for creating VectorStore instances from settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.domain.interfaces import VectorStore
from rag.vectorstore.chroma_store import ChromaVectorStore
from rag.vectorstore.faiss_store import FaissVectorStore

if TYPE_CHECKING:
    from rag.config import Settings


class VectorStoreFactory:
    """Instantiates a VectorStore according to application configuration."""

    @staticmethod
    def create(settings: Settings) -> VectorStore:
        """Create and return a VectorStore matching ``settings.vector_store_backend``."""
        backend = settings.vector_store_backend.lower()

        if backend == "faiss":
            return FaissVectorStore(
                dimension=settings.vector_dimension,
                index_dir=settings.vector_index_dir,
                index_type=settings.faiss_index_type,
                m=settings.faiss_hnsw_m,
                ef_search=settings.faiss_hnsw_ef_search,
                ef_construction=settings.faiss_hnsw_ef_construction,
            )
        if backend == "chroma":
            return ChromaVectorStore(
                dimension=settings.vector_dimension,
                index_dir=settings.vector_index_dir,
            )
        raise ValueError(f"Unknown vector store backend: {settings.vector_store_backend}")
