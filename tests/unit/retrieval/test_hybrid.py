"""Unit tests for HybridRetriever — written first (TDD)."""

from __future__ import annotations

import uuid
from pathlib import Path

from rag.domain.entities import (
    Chunk,
    RetrievalFilter,
    RetrievalResult,
    RetrievedChunk,
    StageTrace,
    VectorRecord,
)
from rag.embeddings.fake import FakeEmbedder
from rag.retrieval.bm25 import Bm25Index
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.fakes import InMemoryChunkLookup
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.sparse import SparseRetriever
from rag.vectorstore.faiss_store import FaissVectorStore


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


class _RecordingRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.requested_ks: list[int] = []

    def retrieve(
        self,
        query: str,
        k: int,
        filter: RetrievalFilter | None = None,
    ) -> RetrievalResult:
        self.requested_ks.append(k)
        return RetrievalResult(
            chunks=self.chunks[:k],
            trace=[StageTrace(stage="test", latency_ms=0.0, results=[])],
        )


class TestHybridRetriever:
    def test_candidate_k_and_fusion_k_bound_output_and_trace(self) -> None:
        chunks = [
            RetrievedChunk(
                chunk_id=uuid.UUID(f"00000000-0000-0000-0000-{i:012d}"),
                document_id=uuid.uuid4(),
                text=f"chunk {i}",
                page=None,
                section=None,
                score=1.0,
                source="dense",
            )
            for i in range(1, 4)
        ]
        dense = _RecordingRetriever(chunks)
        sparse = _RecordingRetriever(chunks)
        retriever = HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            candidate_k=3,
            fusion_k=1,
        )

        result = retriever.retrieve(query="query", k=3)

        assert dense.requested_ks == [3]
        assert sparse.requested_ks == [3]
        assert len(result.chunks) == 1
        assert len(result.trace[-1].results) == 1

    def test_chunk_only_in_dense_and_chunk_only_in_sparse_both_appear_in_fused(
        self, tmp_path: Path
    ) -> None:
        dim = 16
        embedder = FakeEmbedder(dimension=dim)
        vector_store = FaissVectorStore(
            dimension=dim, index_dir=tmp_path / "faiss", index_type="flat"
        )
        sparse_index = Bm25Index()
        chunk_lookup = InMemoryChunkLookup()

        doc_id = uuid.uuid4()
        cid_dense_only = uuid.uuid4()
        cid_sparse_only = uuid.uuid4()

        # Chunk 1: Indexed ONLY in dense vector store
        c_dense = _make_chunk(
            doc_id,
            cid_dense_only,
            "Dense semantic content about artificial neural networks",
        )
        chunk_lookup.add(c_dense)
        v_dense = embedder.embed_passages([c_dense.text])[0]
        vector_store.add(
            [
                VectorRecord(
                    chunk_id=cid_dense_only,
                    vector=v_dense,
                    document_id=doc_id,
                    metadata={"document_id": str(doc_id)},
                )
            ]
        )

        # Chunk 2: Indexed ONLY in sparse BM25 index
        c_sparse = _make_chunk(
            doc_id,
            cid_sparse_only,
            "Exact keyword error code ERR-9942 failure scenario",
        )
        chunk_lookup.add(c_sparse)
        sparse_index.add([c_sparse])

        dense_retriever = DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            chunk_lookup=chunk_lookup,
        )
        sparse_retriever = SparseRetriever(
            index=sparse_index,
            chunk_lookup=chunk_lookup,
        )

        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            candidate_k=50,
            fusion_k=10,
        )

        # Query matches dense semantically and sparse on keyword
        result = hybrid_retriever.retrieve(query="neural networks ERR-9942", k=10)

        retrieved_ids = [c.chunk_id for c in result.chunks]
        assert cid_dense_only in retrieved_ids
        assert cid_sparse_only in retrieved_ids
        for c in result.chunks:
            assert c.source == "fused"

        # Check stage traces
        stages = [t.stage for t in result.trace]
        assert stages == ["dense", "sparse", "fused"]
        for t in result.trace:
            assert t.latency_ms >= 0.0

    def test_hybrid_filter_propagates_to_dense_and_sparse(self, tmp_path: Path) -> None:
        dim = 16
        embedder = FakeEmbedder(dimension=dim)
        vector_store = FaissVectorStore(
            dimension=dim, index_dir=tmp_path / "faiss", index_type="flat"
        )
        sparse_index = Bm25Index()
        chunk_lookup = InMemoryChunkLookup()

        doc_allowed = uuid.uuid4()
        doc_blocked = uuid.uuid4()

        cid_allowed = uuid.uuid4()
        cid_blocked = uuid.uuid4()

        c_allowed = _make_chunk(doc_allowed, cid_allowed, "Common database index topic")
        c_blocked = _make_chunk(doc_blocked, cid_blocked, "Common database index topic")

        chunk_lookup.add_many([c_allowed, c_blocked])
        sparse_index.add([c_allowed, c_blocked])

        v_allowed = embedder.embed_passages([c_allowed.text])[0]
        v_blocked = embedder.embed_passages([c_blocked.text])[0]
        vector_store.add(
            [
                VectorRecord(
                    chunk_id=cid_allowed,
                    vector=v_allowed,
                    document_id=doc_allowed,
                    metadata={"document_id": str(doc_allowed)},
                ),
                VectorRecord(
                    chunk_id=cid_blocked,
                    vector=v_blocked,
                    document_id=doc_blocked,
                    metadata={"document_id": str(doc_blocked)},
                ),
            ]
        )

        dense_retriever = DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            chunk_lookup=chunk_lookup,
        )
        sparse_retriever = SparseRetriever(
            index=sparse_index,
            chunk_lookup=chunk_lookup,
        )

        hybrid = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )

        filter_obj = RetrievalFilter(document_ids=[doc_allowed])
        result = hybrid.retrieve(query="database index", k=5, filter=filter_obj)

        assert len(result.chunks) == 1
        assert result.chunks[0].chunk_id == cid_allowed
        assert result.chunks[0].document_id == doc_allowed

    def test_hybrid_empty_index_and_query(self, tmp_path: Path) -> None:
        dim = 16
        embedder = FakeEmbedder(dimension=dim)
        vector_store = FaissVectorStore(
            dimension=dim, index_dir=tmp_path / "faiss", index_type="flat"
        )
        sparse_index = Bm25Index()
        chunk_lookup = InMemoryChunkLookup()

        dense = DenseRetriever(embedder, vector_store, chunk_lookup)
        sparse = SparseRetriever(sparse_index, chunk_lookup)
        hybrid = HybridRetriever(dense, sparse)

        result_empty_query = hybrid.retrieve(query="   ", k=5)
        assert result_empty_query.chunks == []
        assert result_empty_query.top_score is None

        result_empty_index = hybrid.retrieve(query="something", k=5)
        assert result_empty_index.chunks == []
        assert result_empty_index.top_score is None
