"""Tests for VectorStoreFactory."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import Settings
from rag.vectorstore.chroma_store import ChromaVectorStore
from rag.vectorstore.factory import VectorStoreFactory
from rag.vectorstore.faiss_store import FaissVectorStore


def test_factory_creates_faiss(tmp_path: Path) -> None:
    settings = Settings(
        vector_store_backend="faiss",
        vector_dimension=128,
        vector_index_dir=str(tmp_path / "faiss_test"),
        faiss_index_type="flat",
    )
    store = VectorStoreFactory.create(settings=settings)
    assert isinstance(store, FaissVectorStore)
    assert store.dimension == 128


def test_factory_creates_chroma(tmp_path: Path) -> None:
    settings = Settings(
        vector_store_backend="chroma",
        vector_dimension=64,
        vector_index_dir=str(tmp_path / "chroma_test"),
    )
    store = VectorStoreFactory.create(settings=settings)
    assert isinstance(store, ChromaVectorStore)
    assert store.dimension == 64


def test_factory_unknown_backend_raises() -> None:
    settings = Settings(vector_store_backend="pinecone_cloud")
    with pytest.raises(ValueError, match="Unknown vector store backend"):
        VectorStoreFactory.create(settings=settings)
