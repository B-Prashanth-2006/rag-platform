"""Unit tests for SparseRetriever — written first (TDD)."""

from __future__ import annotations

import uuid

from rag.domain.entities import Chunk, RetrievalFilter
from rag.retrieval.bm25 import Bm25Index
from rag.retrieval.fakes import InMemoryChunkLookup
from rag.retrieval.sparse import SparseRetriever


def _make_chunk(
    doc_id: uuid.UUID,
    chunk_id: uuid.UUID,
    text: str,
    page: int | None = 1,
    section: str | None = "Section",
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


class TestSparseRetriever:
    def test_sparse_retrieval_and_hydration(self) -> None:
        index = Bm25Index()
        chunk_lookup = InMemoryChunkLookup()

        doc_id = uuid.uuid4()
        cid1 = uuid.uuid4()
        cid2 = uuid.uuid4()

        c1 = _make_chunk(doc_id, cid1, "Kubernetes ingress controller configuration")
        c2 = _make_chunk(doc_id, cid2, "PostgreSQL high availability clustering")

        index.add([c1, c2])
        chunk_lookup.add(c1)
        chunk_lookup.add(c2)

        retriever = SparseRetriever(index=index, chunk_lookup=chunk_lookup)

        result = retriever.retrieve(query="Kubernetes ingress", k=2)

        assert len(result.chunks) >= 1
        assert result.chunks[0].chunk_id == cid1
        assert result.chunks[0].text == "Kubernetes ingress controller configuration"
        assert result.chunks[0].source == "sparse"
        assert result.top_score is not None

        # Verify trace
        assert len(result.trace) == 1
        assert result.trace[0].stage == "sparse"
        assert result.trace[0].latency_ms >= 0.0

    def test_sparse_retrieval_with_document_filter(self) -> None:
        index = Bm25Index()
        chunk_lookup = InMemoryChunkLookup()

        doc_target = uuid.uuid4()
        doc_other = uuid.uuid4()

        cid_target = uuid.uuid4()
        cid_other = uuid.uuid4()

        c_target = _make_chunk(doc_target, cid_target, "Docker container networking guide")
        c_other = _make_chunk(doc_other, cid_other, "Docker volumes and storage drivers")

        index.add([c_target, c_other])
        chunk_lookup.add(c_target)
        chunk_lookup.add(c_other)

        retriever = SparseRetriever(index=index, chunk_lookup=chunk_lookup)

        result = retriever.retrieve(
            query="Docker",
            k=5,
            filter=RetrievalFilter(document_ids=[doc_target]),
        )

        assert len(result.chunks) == 1
        assert result.chunks[0].chunk_id == cid_target
        assert result.chunks[0].document_id == doc_target

    def test_sparse_retrieval_empty_query_returns_empty_result(self) -> None:
        index = Bm25Index()
        chunk_lookup = InMemoryChunkLookup()
        c = _make_chunk(uuid.uuid4(), uuid.uuid4(), "Some test chunk")
        index.add([c])
        chunk_lookup.add(c)

        retriever = SparseRetriever(index=index, chunk_lookup=chunk_lookup)
        result = retriever.retrieve(query="", k=5)

        assert result.chunks == []
        assert result.top_score is None
        assert len(result.trace) == 1
        assert result.trace[0].stage == "sparse"
