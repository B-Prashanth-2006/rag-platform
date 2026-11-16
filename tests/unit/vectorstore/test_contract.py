"""Contract test suite shared by all VectorStore backends."""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from rag.domain.entities import SearchHit, VectorRecord
from rag.domain.interfaces import VectorStore
from rag.vectorstore.chroma_store import ChromaVectorStore
from rag.vectorstore.faiss_store import FaissVectorStore


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return [1.0] + [0.0] * (len(vec) - 1)
    return [x / norm for x in vec]


@pytest.fixture(params=["faiss_flat", "faiss_hnsw", "chroma"])
def store_setup(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> tuple[VectorStore, Callable[[], VectorStore]]:
    backend = request.param
    dim = 4

    if backend == "faiss_flat":
        store_dir = tmp_path / "faiss_flat"

        def factory() -> VectorStore:
            return FaissVectorStore(
                dimension=dim,
                index_dir=store_dir,
                index_type="flat",
            )

    elif backend == "faiss_hnsw":
        store_dir = tmp_path / "faiss_hnsw"

        def factory() -> VectorStore:
            return FaissVectorStore(
                dimension=dim,
                index_dir=store_dir,
                index_type="hnsw",
                m=16,
                ef_search=32,
                ef_construction=32,
            )

    elif backend == "chroma":
        store_dir = tmp_path / "chroma"

        def factory() -> VectorStore:
            return ChromaVectorStore(
                dimension=dim,
                index_dir=store_dir,
                collection_name="test_contract",
            )
    else:
        raise ValueError(f"Unknown backend fixture: {backend}")

    return factory(), factory


@pytest.fixture
def store(store_setup: tuple[VectorStore, Callable[[], VectorStore]]) -> VectorStore:
    return store_setup[0]


def test_satisfies_protocol(store: VectorStore) -> None:
    assert isinstance(store, VectorStore)


def test_empty_store_returns_empty(store: VectorStore) -> None:
    assert store.count() == 0
    query = _normalize([1.0, 0.0, 0.0, 0.0])
    assert store.search(query, k=5) == []


def test_add_and_search_exact_match(store: VectorStore) -> None:
    doc_id = uuid.uuid4()
    c1 = uuid.uuid4()
    c2 = uuid.uuid4()

    v1 = _normalize([1.0, 0.0, 0.0, 0.0])
    v2 = _normalize([0.0, 1.0, 0.0, 0.0])

    r1 = VectorRecord(chunk_id=c1, vector=v1, document_id=doc_id, metadata={"tag": "first"})
    r2 = VectorRecord(chunk_id=c2, vector=v2, document_id=doc_id, metadata={"tag": "second"})

    store.add([r1, r2])
    assert store.count() == 2

    # Query with exact vector v1
    hits = store.search(v1, k=2)
    assert len(hits) == 2
    assert isinstance(hits[0], SearchHit)
    assert hits[0].chunk_id == c1
    # Near 1.0 similarity for exact vector
    assert math.isclose(hits[0].score, 1.0, abs_tol=1e-3)
    assert hits[1].chunk_id == c2
    assert hits[1].score < hits[0].score


def test_k_larger_than_store_returns_all(store: VectorStore) -> None:
    doc_id = uuid.uuid4()
    records = [
        VectorRecord(
            chunk_id=uuid.uuid4(),
            vector=_normalize([float(i), 1.0, 0.0, 0.0]),
            document_id=doc_id,
            metadata={"i": i},
        )
        for i in range(3)
    ]
    store.add(records)

    hits = store.search(_normalize([1.0, 1.0, 0.0, 0.0]), k=10)
    assert len(hits) == 3


def test_results_sorted_descending_by_score_and_tie_breaking(store: VectorStore) -> None:
    doc_id = uuid.uuid4()
    # Create two identical vectors with different chunk_ids to test tie-breaking
    c_low = uuid.UUID("00000000-0000-0000-0000-000000000001")
    c_high = uuid.UUID("00000000-0000-0000-0000-000000000002")
    c_diff = uuid.UUID("00000000-0000-0000-0000-000000000003")

    v_same = _normalize([1.0, 0.0, 0.0, 0.0])
    v_diff = _normalize([0.0, 0.0, 1.0, 0.0])

    r_high = VectorRecord(chunk_id=c_high, vector=v_same, document_id=doc_id, metadata={})
    r_low = VectorRecord(chunk_id=c_low, vector=v_same, document_id=doc_id, metadata={})
    r_diff = VectorRecord(chunk_id=c_diff, vector=v_diff, document_id=doc_id, metadata={})

    store.add([r_high, r_low, r_diff])
    hits = store.search(v_same, k=3)

    assert len(hits) == 3
    # Both identical vectors have the highest score
    assert math.isclose(hits[0].score, hits[1].score, abs_tol=1e-4)
    # Tie broken by chunk_id string order
    assert hits[0].chunk_id == c_low
    assert hits[1].chunk_id == c_high
    assert hits[2].chunk_id == c_diff
    assert hits[2].score < hits[1].score


def test_metadata_filtering(store: VectorStore) -> None:
    doc1 = uuid.uuid4()
    doc2 = uuid.uuid4()
    c1 = uuid.uuid4()
    c2 = uuid.uuid4()
    c3 = uuid.uuid4()

    v = _normalize([1.0, 0.0, 0.0, 0.0])

    r1 = VectorRecord(chunk_id=c1, vector=v, document_id=doc1, metadata={"category": "A"})
    r2 = VectorRecord(chunk_id=c2, vector=v, document_id=doc1, metadata={"category": "B"})
    r3 = VectorRecord(chunk_id=c3, vector=v, document_id=doc2, metadata={"category": "A"})

    store.add([r1, r2, r3])

    # Filter by document_id
    hits_doc1 = store.search(v, k=5, filter={"document_id": str(doc1)})
    assert len(hits_doc1) == 2
    assert {h.chunk_id for h in hits_doc1} == {c1, c2}

    # Filter by metadata attribute
    hits_cat_a = store.search(v, k=5, filter={"category": "A"})
    assert len(hits_cat_a) == 2
    assert {h.chunk_id for h in hits_cat_a} == {c1, c3}


def test_delete_chunks(store: VectorStore) -> None:
    doc_id = uuid.uuid4()
    c1 = uuid.uuid4()
    c2 = uuid.uuid4()

    v = _normalize([1.0, 0.0, 0.0, 0.0])
    store.add(
        [
            VectorRecord(chunk_id=c1, vector=v, document_id=doc_id, metadata={}),
            VectorRecord(chunk_id=c2, vector=v, document_id=doc_id, metadata={}),
        ]
    )
    assert store.count() == 2

    # Delete c1 and a non-existent chunk (should be no-op for missing id)
    missing = uuid.uuid4()
    store.delete_chunks([c1, missing])
    assert store.count() == 1

    hits = store.search(v, k=5)
    assert len(hits) == 1
    assert hits[0].chunk_id == c2


def test_delete_document_removes_only_that_document(store: VectorStore) -> None:
    doc1 = uuid.uuid4()
    doc2 = uuid.uuid4()
    c1 = uuid.uuid4()
    c2 = uuid.uuid4()
    c3 = uuid.uuid4()

    v = _normalize([1.0, 0.0, 0.0, 0.0])
    store.add(
        [
            VectorRecord(chunk_id=c1, vector=v, document_id=doc1, metadata={}),
            VectorRecord(chunk_id=c2, vector=v, document_id=doc1, metadata={}),
            VectorRecord(chunk_id=c3, vector=v, document_id=doc2, metadata={}),
        ]
    )
    assert store.count() == 3

    store.delete_document(doc1)
    assert store.count() == 1

    hits = store.search(v, k=5)
    assert len(hits) == 1
    assert hits[0].chunk_id == c3


def test_replace_on_readd_upsert(store: VectorStore) -> None:
    doc_id = uuid.uuid4()
    c1 = uuid.uuid4()

    v_old = _normalize([1.0, 0.0, 0.0, 0.0])
    v_new = _normalize([0.0, 1.0, 0.0, 0.0])

    store.add([VectorRecord(chunk_id=c1, vector=v_old, document_id=doc_id, metadata={"v": "old"})])
    assert store.count() == 1

    # Re-add with same chunk_id, different vector and metadata
    store.add([VectorRecord(chunk_id=c1, vector=v_new, document_id=doc_id, metadata={"v": "new"})])
    assert store.count() == 1

    # Search with v_new should match c1 with high score
    hits = store.search(v_new, k=1)
    assert len(hits) == 1
    assert hits[0].chunk_id == c1
    assert math.isclose(hits[0].score, 1.0, abs_tol=1e-3)


def test_persist_and_reload(
    store_setup: tuple[VectorStore, Callable[[], VectorStore]],
) -> None:
    store, reloader = store_setup
    doc_id = uuid.uuid4()
    c1 = uuid.uuid4()
    c2 = uuid.uuid4()

    v1 = _normalize([1.0, 0.0, 0.0, 0.0])
    v2 = _normalize([0.0, 1.0, 0.0, 0.0])

    store.add(
        [
            VectorRecord(chunk_id=c1, vector=v1, document_id=doc_id, metadata={"key": "val1"}),
            VectorRecord(chunk_id=c2, vector=v2, document_id=doc_id, metadata={"key": "val2"}),
        ]
    )
    store.persist()

    # Load new store instance from same directory
    reloaded = reloader()
    assert reloaded.count() == 2

    hits_v1 = reloaded.search(v1, k=2)
    assert len(hits_v1) == 2
    assert hits_v1[0].chunk_id == c1
    assert math.isclose(hits_v1[0].score, 1.0, abs_tol=1e-3)

    # Filtering works on reloaded store
    filtered = reloaded.search(v1, k=2, filter={"key": "val2"})
    assert len(filtered) == 1
    assert filtered[0].chunk_id == c2
