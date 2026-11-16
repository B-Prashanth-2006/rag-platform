"""Unit tests for Bm25Index (sparse index) — written first (TDD)."""

from __future__ import annotations

import uuid

from rag.domain.entities import Chunk, RetrievalFilter
from rag.retrieval.bm25 import Bm25Index


def _make_chunk(
    text: str,
    doc_id: uuid.UUID | None = None,
    chunk_id: uuid.UUID | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id or uuid.uuid4(),
        document_id=doc_id or uuid.uuid4(),
        index=0,
        text=text,
        page=1,
        section=None,
        token_count=len(text.split()),
        embedding_model_version=None,
    )


class TestBm25Index:
    def test_exact_term_query_ranks_matching_chunk_first(self) -> None:
        c1 = _make_chunk("General introduction to machine learning principles.")
        c2 = _make_chunk("Critical database outage occurred with error code ERR-4521.")
        c3 = _make_chunk("API performance optimization and network caching guide.")

        index = Bm25Index()
        index.add([c1, c2, c3])

        hits = index.search(query="ERR-4521", k=3)
        assert len(hits) > 0
        top_chunk_id, top_score = hits[0]
        assert top_chunk_id == c2.id
        assert top_score > 0.0

    def test_filter_by_document(self) -> None:
        doc1 = uuid.uuid4()
        doc2 = uuid.uuid4()

        c1 = _make_chunk("PostgreSQL query tuning and indexing techniques.", doc_id=doc1)
        c2 = _make_chunk("PostgreSQL connection pooling with PgBouncer.", doc_id=doc2)

        index = Bm25Index()
        index.add([c1, c2])

        # Filter strictly for doc1
        hits = index.search(
            query="PostgreSQL",
            k=5,
            filter=RetrievalFilter(document_ids=[doc1]),
        )
        assert len(hits) == 1
        assert hits[0][0] == c1.id

    def test_delete_document(self) -> None:
        doc1 = uuid.uuid4()
        doc2 = uuid.uuid4()

        c1 = _make_chunk("First document content text.", doc_id=doc1)
        c2 = _make_chunk("Second document content text.", doc_id=doc2)

        index = Bm25Index()
        index.add([c1, c2])
        assert index.count() == 2

        index.delete_document(doc1)
        assert index.count() == 1

        hits = index.search(query="document content", k=5)
        assert len(hits) == 1
        assert hits[0][0] == c2.id

    def test_empty_index_or_empty_query_returns_empty(self) -> None:
        index = Bm25Index()
        assert index.search("query", k=5) == []

        c = _make_chunk("Some valid content.")
        index.add([c])
        assert index.search("", k=5) == []
        assert index.search("   ", k=5) == []

    def test_deterministic_tie_breaking(self) -> None:
        cid1 = uuid.UUID("00000000-0000-0000-0000-000000000001")
        cid2 = uuid.UUID("00000000-0000-0000-0000-000000000002")

        c1 = _make_chunk("exact identical match", chunk_id=cid2)
        c2 = _make_chunk("exact identical match", chunk_id=cid1)

        index = Bm25Index()
        index.add([c1, c2])

        hits = index.search("exact identical match", k=2)
        assert len(hits) == 2
        # Same score, tie-break by chunk_id ascending
        assert hits[0][0] == cid1
        assert hits[1][0] == cid2
