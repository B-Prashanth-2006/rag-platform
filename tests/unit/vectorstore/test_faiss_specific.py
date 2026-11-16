"""Tests for FAISS-specific behavior: dimension checks, tombstoning, compaction, atomic writes."""

from __future__ import annotations

import math
import uuid
from pathlib import Path

import pytest

from rag.domain.entities import VectorRecord
from rag.domain.errors import DimensionMismatchError
from rag.vectorstore.faiss_store import FaissVectorStore


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return [1.0] + [0.0] * (len(vec) - 1)
    return [x / norm for x in vec]


def test_faiss_dimension_mismatch_raises(tmp_path: Path) -> None:
    store = FaissVectorStore(dimension=4, index_dir=tmp_path / "faiss", index_type="flat")
    doc_id = uuid.uuid4()

    # Wrong dimension on add
    record_5d = VectorRecord(
        chunk_id=uuid.uuid4(),
        vector=[1.0, 0.0, 0.0, 0.0, 0.0],
        document_id=doc_id,
        metadata={},
    )
    with pytest.raises(DimensionMismatchError) as exc_info:
        store.add([record_5d])
    assert exc_info.value.expected == 4
    assert exc_info.value.got == 5

    # Wrong dimension on search
    with pytest.raises(DimensionMismatchError) as exc_info_search:
        store.search([1.0, 0.0, 0.0], k=3)
    assert exc_info_search.value.expected == 4
    assert exc_info_search.value.got == 3


def test_faiss_hnsw_tombstones_excluded_from_search(tmp_path: Path) -> None:
    store = FaissVectorStore(
        dimension=4,
        index_dir=tmp_path / "faiss_hnsw",
        index_type="hnsw",
        m=16,
        ef_search=32,
        ef_construction=32,
    )
    doc_id = uuid.uuid4()
    c1 = uuid.uuid4()
    c2 = uuid.uuid4()
    v1 = _normalize([1.0, 0.0, 0.0, 0.0])
    v2 = _normalize([0.0, 1.0, 0.0, 0.0])

    store.add(
        [
            VectorRecord(chunk_id=c1, vector=v1, document_id=doc_id, metadata={}),
            VectorRecord(chunk_id=c2, vector=v2, document_id=doc_id, metadata={}),
        ]
    )
    assert store.count() == 2

    # Delete c1 (tombstones it in HNSW)
    store.delete_chunks([c1])
    assert store.count() == 1

    # Query directly with v1: c1 was deleted, so it must NOT appear
    hits = store.search(v1, k=5)
    assert len(hits) == 1
    assert hits[0].chunk_id == c2


def test_faiss_hnsw_compact_rebuilds_index(tmp_path: Path) -> None:
    store = FaissVectorStore(
        dimension=4,
        index_dir=tmp_path / "faiss_hnsw",
        index_type="hnsw",
        m=16,
        ef_search=32,
        ef_construction=32,
    )
    doc_id = uuid.uuid4()
    records = [
        VectorRecord(
            chunk_id=uuid.uuid4(),
            vector=_normalize([float(i), 1.0, 0.0, 0.0]),
            document_id=doc_id,
            metadata={"i": i},
        )
        for i in range(10)
    ]
    store.add(records)
    assert store.count() == 10

    # Delete first 4 chunks
    deleted_ids = [r.chunk_id for r in records[:4]]
    store.delete_chunks(deleted_ids)
    assert store.count() == 6

    # Call compact() to rebuild
    store.compact()
    assert store.count() == 6

    # All deleted chunks are gone
    query = _normalize([0.0, 1.0, 0.0, 0.0])
    hits = store.search(query, k=10)
    hit_ids = {h.chunk_id for h in hits}
    for did in deleted_ids:
        assert did not in hit_ids


def test_faiss_atomic_persistence(tmp_path: Path) -> None:
    index_dir = tmp_path / "faiss_atomic"
    store = FaissVectorStore(dimension=4, index_dir=index_dir, index_type="flat")
    doc_id = uuid.uuid4()
    c1 = uuid.uuid4()
    v1 = _normalize([1.0, 0.0, 0.0, 0.0])
    store.add([VectorRecord(chunk_id=c1, vector=v1, document_id=doc_id, metadata={})])
    store.persist()

    # Simulate a leftover temp file from an interrupted write
    temp_file = index_dir / "index.faiss.tmp_12345"
    temp_file.write_text("corrupted partial data")

    # Reloading from index_dir must succeed and load valid state
    reloaded = FaissVectorStore(dimension=4, index_dir=index_dir, index_type="flat")
    assert reloaded.count() == 1
    hits = reloaded.search(v1, k=1)
    assert len(hits) == 1
    assert hits[0].chunk_id == c1
