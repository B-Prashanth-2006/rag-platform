"""Unit tests for RetrievalPipeline and build_retriever — written first (TDD)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TypedDict

import pytest

from rag.config import Settings
from rag.domain.entities import Chunk, RetrievalFilter, VectorRecord
from rag.embeddings.fake import FakeEmbedder
from rag.retrieval.bm25 import Bm25Index
from rag.retrieval.fakes import FakeReranker, InMemoryChunkLookup
from rag.retrieval.pipeline import build_retriever
from rag.vectorstore.faiss_store import FaissVectorStore


class SetupComponents(TypedDict):
    embedder: FakeEmbedder
    vector_store: FaissVectorStore
    sparse_index: Bm25Index
    chunk_lookup: InMemoryChunkLookup
    reranker: FakeReranker
    c1: Chunk
    c2: Chunk
    doc1: uuid.UUID
    doc2: uuid.UUID


def _make_chunk(
    doc_id: uuid.UUID,
    chunk_id: uuid.UUID,
    text: str,
    page: int | None = 1,
    section: str | None = "Intro",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=doc_id,
        index=0,
        text=text,
        page=page,
        section=section,
        token_count=len(text.split()),
        embedding_model_version=None,
    )


class TestRetrievalPipeline:
    @pytest.fixture
    def setup_components(self, tmp_path: Path) -> SetupComponents:
        dim = 16
        embedder = FakeEmbedder(dimension=dim)
        vector_store = FaissVectorStore(
            dimension=dim, index_dir=tmp_path / "faiss", index_type="flat"
        )
        sparse_index = Bm25Index()
        chunk_lookup = InMemoryChunkLookup()

        doc1 = uuid.uuid4()
        doc2 = uuid.uuid4()
        c1 = _make_chunk(doc1, uuid.uuid4(), "Machine learning algorithms and optimization")
        c2 = _make_chunk(doc2, uuid.uuid4(), "Database index structure and SQL queries")

        chunk_lookup.add_many([c1, c2])
        sparse_index.add([c1, c2])

        v1 = embedder.embed_passages([c1.text])[0]
        v2 = embedder.embed_passages([c2.text])[0]
        vector_store.add(
            [
                VectorRecord(
                    chunk_id=c1.id,
                    vector=v1,
                    document_id=doc1,
                    metadata={"document_id": str(doc1)},
                ),
                VectorRecord(
                    chunk_id=c2.id,
                    vector=v2,
                    document_id=doc2,
                    metadata={"document_id": str(doc2)},
                ),
            ]
        )

        fake_reranker = FakeReranker(
            score_fn=lambda q, c: 0.99 if "machine" in c.text.lower() else 0.1
        )

        return {
            "embedder": embedder,
            "vector_store": vector_store,
            "sparse_index": sparse_index,
            "chunk_lookup": chunk_lookup,
            "reranker": fake_reranker,
            "c1": c1,
            "c2": c2,
            "doc1": doc1,
            "doc2": doc2,
        }

    def test_dense_mode_without_reranker(self, setup_components: SetupComponents) -> None:
        c = setup_components
        settings = Settings(retrieval_mode="dense", use_reranker=False)
        pipeline = build_retriever(
            settings=settings,
            embedder=c["embedder"],
            vector_store=c["vector_store"],
            sparse_index=c["sparse_index"],
            chunk_lookup=c["chunk_lookup"],
        )

        result = pipeline.retrieve(query="machine learning", k=2)
        assert len(result.chunks) > 0
        stages = [t.stage for t in result.trace]
        assert stages == ["dense"]

    def test_dense_mode_with_reranker(self, setup_components: SetupComponents) -> None:
        c = setup_components
        settings = Settings(retrieval_mode="dense", use_reranker=True)
        pipeline = build_retriever(
            settings=settings,
            embedder=c["embedder"],
            vector_store=c["vector_store"],
            sparse_index=c["sparse_index"],
            chunk_lookup=c["chunk_lookup"],
            reranker=c["reranker"],
        )

        result = pipeline.retrieve(query="machine learning", k=2)
        assert len(result.chunks) > 0
        stages = [t.stage for t in result.trace]
        assert stages == ["dense", "reranked"]
        assert result.chunks[0].source == "reranked"

    def test_sparse_mode_without_reranker(self, setup_components: SetupComponents) -> None:
        c = setup_components
        settings = Settings(retrieval_mode="sparse", use_reranker=False)
        pipeline = build_retriever(
            settings=settings,
            embedder=c["embedder"],
            vector_store=c["vector_store"],
            sparse_index=c["sparse_index"],
            chunk_lookup=c["chunk_lookup"],
        )

        result = pipeline.retrieve(query="database SQL", k=2)
        assert len(result.chunks) > 0
        stages = [t.stage for t in result.trace]
        assert stages == ["sparse"]

    def test_sparse_mode_with_reranker(self, setup_components: SetupComponents) -> None:
        c = setup_components
        settings = Settings(retrieval_mode="sparse", use_reranker=True)
        pipeline = build_retriever(
            settings=settings,
            embedder=c["embedder"],
            vector_store=c["vector_store"],
            sparse_index=c["sparse_index"],
            chunk_lookup=c["chunk_lookup"],
            reranker=c["reranker"],
        )

        result = pipeline.retrieve(query="database SQL", k=2)
        assert len(result.chunks) > 0
        stages = [t.stage for t in result.trace]
        assert stages == ["sparse", "reranked"]
        assert result.chunks[0].source == "reranked"

    def test_hybrid_mode_without_reranker(self, setup_components: SetupComponents) -> None:
        c = setup_components
        settings = Settings(retrieval_mode="hybrid", use_reranker=False)
        pipeline = build_retriever(
            settings=settings,
            embedder=c["embedder"],
            vector_store=c["vector_store"],
            sparse_index=c["sparse_index"],
            chunk_lookup=c["chunk_lookup"],
        )

        result = pipeline.retrieve(query="database machine", k=2)
        assert len(result.chunks) > 0
        stages = [t.stage for t in result.trace]
        assert stages == ["dense", "sparse", "fused"]

    def test_hybrid_mode_with_reranker(self, setup_components: SetupComponents) -> None:
        c = setup_components
        settings = Settings(retrieval_mode="hybrid", use_reranker=True)
        pipeline = build_retriever(
            settings=settings,
            embedder=c["embedder"],
            vector_store=c["vector_store"],
            sparse_index=c["sparse_index"],
            chunk_lookup=c["chunk_lookup"],
            reranker=c["reranker"],
        )

        result = pipeline.retrieve(query="database machine", k=2)
        assert len(result.chunks) > 0
        stages = [t.stage for t in result.trace]
        assert stages == ["dense", "sparse", "fused", "reranked"]
        assert result.chunks[0].source == "reranked"

    def test_unknown_retrieval_mode_raises_value_error(
        self, setup_components: SetupComponents
    ) -> None:
        c = setup_components
        settings = Settings(retrieval_mode="quantum")
        with pytest.raises(ValueError, match="Unknown retrieval mode"):
            build_retriever(
                settings=settings,
                embedder=c["embedder"],
                vector_store=c["vector_store"],
                sparse_index=c["sparse_index"],
                chunk_lookup=c["chunk_lookup"],
            )

    def test_pipeline_filter_respected(self, setup_components: SetupComponents) -> None:
        c = setup_components
        settings = Settings(retrieval_mode="hybrid", use_reranker=True)
        pipeline = build_retriever(
            settings=settings,
            embedder=c["embedder"],
            vector_store=c["vector_store"],
            sparse_index=c["sparse_index"],
            chunk_lookup=c["chunk_lookup"],
            reranker=c["reranker"],
        )

        filter_obj = RetrievalFilter(document_ids=[c["doc1"]])
        result = pipeline.retrieve(query="database machine", k=2, filter=filter_obj)

        assert len(result.chunks) == 1
        assert result.chunks[0].document_id == c["doc1"]

    def test_pipeline_empty_index_top_score_none(self, tmp_path: Path) -> None:
        dim = 16
        embedder = FakeEmbedder(dimension=dim)
        vector_store = FaissVectorStore(
            dimension=dim, index_dir=tmp_path / "faiss", index_type="flat"
        )
        sparse_index = Bm25Index()
        chunk_lookup = InMemoryChunkLookup()

        settings = Settings(retrieval_mode="hybrid", use_reranker=True)
        pipeline = build_retriever(
            settings=settings,
            embedder=embedder,
            vector_store=vector_store,
            sparse_index=sparse_index,
            chunk_lookup=chunk_lookup,
            reranker=FakeReranker(),
        )

        result = pipeline.retrieve(query="nonexistent", k=5)
        assert result.chunks == []
        assert result.top_score is None
        stages = [t.stage for t in result.trace]
        assert stages == ["dense", "sparse", "fused", "reranked"]
