"""Unit tests for retrieval domain entities and protocols — written first (TDD)."""

from __future__ import annotations

import uuid

import pytest

from rag.domain.entities import (
    RetrievalFilter,
    RetrievalResult,
    RetrievedChunk,
    StageTrace,
)
from rag.domain.interfaces import (
    ChunkLookup,
    Reranker,
    Retriever,
    SparseIndex,
)


class TestRetrievalEntities:
    def test_retrieval_filter(self) -> None:
        doc_id1 = uuid.uuid4()
        doc_id2 = uuid.uuid4()
        rf = RetrievalFilter(document_ids=[doc_id1, doc_id2])
        assert rf.document_ids == [doc_id1, doc_id2]

        rf_empty = RetrievalFilter()
        assert rf_empty.document_ids is None

    def test_retrieved_chunk_creation(self) -> None:
        cid = uuid.uuid4()
        did = uuid.uuid4()
        rc = RetrievedChunk(
            chunk_id=cid,
            document_id=did,
            text="chunk text",
            page=2,
            section="Intro",
            score=0.88,
            source="dense",
        )
        assert rc.chunk_id == cid
        assert rc.document_id == did
        assert rc.text == "chunk text"
        assert rc.page == 2
        assert rc.section == "Intro"
        assert rc.score == 0.88
        assert rc.source == "dense"

    def test_retrieved_chunk_invalid_source_raises(self) -> None:
        cid = uuid.uuid4()
        did = uuid.uuid4()
        with pytest.raises(ValueError, match="Invalid source"):
            RetrievedChunk(
                chunk_id=cid,
                document_id=did,
                text="chunk text",
                page=None,
                section=None,
                score=0.5,
                source="invalid_source",
            )

    def test_stage_trace_creation(self) -> None:
        cid1 = uuid.uuid4()
        cid2 = uuid.uuid4()
        trace = StageTrace(
            stage="dense",
            latency_ms=12.5,
            results=[(cid1, 0.9), (cid2, 0.7)],
        )
        assert trace.stage == "dense"
        assert trace.latency_ms == 12.5
        assert trace.results == [(cid1, 0.9), (cid2, 0.7)]

    def test_retrieval_result_with_top_score(self) -> None:
        cid1 = uuid.uuid4()
        cid2 = uuid.uuid4()
        did = uuid.uuid4()
        c1 = RetrievedChunk(
            chunk_id=cid1,
            document_id=did,
            text="top chunk",
            page=1,
            section=None,
            score=0.95,
            source="dense",
        )
        c2 = RetrievedChunk(
            chunk_id=cid2,
            document_id=did,
            text="second chunk",
            page=1,
            section=None,
            score=0.80,
            source="dense",
        )
        res = RetrievalResult(chunks=[c1, c2], trace=[])
        assert res.top_score == 0.95

    def test_retrieval_result_empty_top_score_none(self) -> None:
        res = RetrievalResult(chunks=[], trace=[])
        assert res.top_score is None


class TestRetrievalProtocols:
    def test_retriever_protocol_conformance(self) -> None:
        class DummyRetriever:
            def retrieve(
                self,
                query: str,
                k: int,
                filter: RetrievalFilter | None = None,
            ) -> RetrievalResult:
                return RetrievalResult(chunks=[], trace=[])

        assert isinstance(DummyRetriever(), Retriever)

    def test_chunk_lookup_protocol_conformance(self) -> None:
        from rag.domain.entities import Chunk

        class DummyChunkLookup:
            def get_many(self, chunk_ids: list[uuid.UUID]) -> list[Chunk]:
                return []

        assert isinstance(DummyChunkLookup(), ChunkLookup)

    def test_sparse_index_protocol_conformance(self) -> None:
        from rag.domain.entities import Chunk

        class DummySparseIndex:
            def add(self, chunks: list[Chunk]) -> None:
                pass

            def search(
                self,
                query: str,
                k: int,
                filter: RetrievalFilter | None = None,
            ) -> list[tuple[uuid.UUID, float]]:
                return []

            def delete_document(self, document_id: uuid.UUID) -> None:
                pass

            def count(self) -> int:
                return 0

        assert isinstance(DummySparseIndex(), SparseIndex)

    def test_reranker_protocol_conformance(self) -> None:
        class DummyReranker:
            def rerank(
                self,
                query: str,
                chunks: list[RetrievedChunk],
                top_k: int,
            ) -> list[RetrievedChunk]:
                return []

        assert isinstance(DummyReranker(), Reranker)
