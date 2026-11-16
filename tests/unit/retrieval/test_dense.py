"""Unit tests for DenseRetriever — written first (TDD)."""

from __future__ import annotations

import uuid
from pathlib import Path

from rag.domain.entities import Chunk, RetrievalFilter, VectorRecord
from rag.embeddings.fake import FakeEmbedder
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.fakes import InMemoryChunkLookup
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


class TestDenseRetriever:
    def test_dense_retrieval_and_hydration(self, tmp_path: Path) -> None:
        dim = 16
        embedder = FakeEmbedder(dimension=dim)
        vector_store = FaissVectorStore(
            dimension=dim, index_dir=tmp_path / "faiss", index_type="flat"
        )
        chunk_lookup = InMemoryChunkLookup()

        doc_id = uuid.uuid4()
        cid1 = uuid.uuid4()
        cid2 = uuid.uuid4()

        c1 = _make_chunk(doc_id, cid1, "Chunk 1 text about apples")
        c2 = _make_chunk(doc_id, cid2, "Chunk 2 text about oranges")
        chunk_lookup.add(c1)
        chunk_lookup.add(c2)

        # Index vectors in vector store
        v1 = embedder.embed_passages([c1.text])[0]
        v2 = embedder.embed_passages([c2.text])[0]
        vector_store.add(
            [
                VectorRecord(
                    chunk_id=cid1,
                    vector=v1,
                    document_id=doc_id,
                    metadata={"document_id": str(doc_id)},
                ),
                VectorRecord(
                    chunk_id=cid2,
                    vector=v2,
                    document_id=doc_id,
                    metadata={"document_id": str(doc_id)},
                ),
            ]
        )

        retriever = DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            chunk_lookup=chunk_lookup,
        )

        result = retriever.retrieve(query="apples", k=2)

        assert len(result.chunks) == 2
        assert result.chunks[0].chunk_id == cid1
        assert result.chunks[0].text == "Chunk 1 text about apples"
        assert result.chunks[0].source == "dense"
        assert result.top_score is not None

        # Verify trace
        assert len(result.trace) == 1
        assert result.trace[0].stage == "dense"
        assert result.trace[0].latency_ms >= 0.0
        assert len(result.trace[0].results) == 2

    def test_dense_retrieval_with_document_filter(self, tmp_path: Path) -> None:
        dim = 16
        embedder = FakeEmbedder(dimension=dim)
        vector_store = FaissVectorStore(
            dimension=dim, index_dir=tmp_path / "faiss", index_type="flat"
        )
        chunk_lookup = InMemoryChunkLookup()

        doc_target = uuid.uuid4()
        doc_other = uuid.uuid4()

        cid_target = uuid.uuid4()
        cid_other = uuid.uuid4()

        c_target = _make_chunk(doc_target, cid_target, "Target document chunk")
        c_other = _make_chunk(doc_other, cid_other, "Other document chunk")
        chunk_lookup.add(c_target)
        chunk_lookup.add(c_other)

        v_target = embedder.embed_passages([c_target.text])[0]
        v_other = embedder.embed_passages([c_other.text])[0]
        vector_store.add(
            [
                VectorRecord(
                    chunk_id=cid_target,
                    vector=v_target,
                    document_id=doc_target,
                    metadata={"document_id": str(doc_target)},
                ),
                VectorRecord(
                    chunk_id=cid_other,
                    vector=v_other,
                    document_id=doc_other,
                    metadata={"document_id": str(doc_other)},
                ),
            ]
        )

        retriever = DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            chunk_lookup=chunk_lookup,
        )

        result = retriever.retrieve(
            query="document chunk",
            k=5,
            filter=RetrievalFilter(document_ids=[doc_target]),
        )

        assert len(result.chunks) == 1
        assert result.chunks[0].chunk_id == cid_target
        assert result.chunks[0].document_id == doc_target

    def test_empty_vector_store_returns_empty_result(self, tmp_path: Path) -> None:
        dim = 16
        embedder = FakeEmbedder(dimension=dim)
        vector_store = FaissVectorStore(
            dimension=dim, index_dir=tmp_path / "faiss", index_type="flat"
        )
        chunk_lookup = InMemoryChunkLookup()

        retriever = DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            chunk_lookup=chunk_lookup,
        )

        result = retriever.retrieve(query="anything", k=10)
        assert result.chunks == []
        assert result.top_score is None
        assert len(result.trace) == 1
        assert result.trace[0].stage == "dense"
