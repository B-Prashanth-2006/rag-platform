"""Cross-backend equivalence tests between FAISS and Chroma."""

from __future__ import annotations

import math
import random
import uuid
from pathlib import Path

from rag.domain.entities import VectorRecord
from rag.vectorstore.chroma_store import ChromaVectorStore
from rag.vectorstore.faiss_store import FaissVectorStore


def _random_normalized_vector(rng: random.Random, dim: int) -> list[float]:
    raw = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def test_faiss_flat_and_chroma_same_top_1(tmp_path: Path) -> None:
    rng = random.Random(42)
    dim = 16
    num_records = 100
    num_queries = 50

    faiss_dir = tmp_path / "faiss"
    chroma_dir = tmp_path / "chroma"

    faiss_store = FaissVectorStore(dimension=dim, index_dir=faiss_dir, index_type="flat")
    chroma_store = ChromaVectorStore(
        dimension=dim,
        index_dir=chroma_dir,
        collection_name="cross_test",
    )

    doc_id = uuid.uuid4()
    records: list[VectorRecord] = []
    for i in range(num_records):
        cid = uuid.UUID(int=i + 1)
        vec = _random_normalized_vector(rng, dim)
        records.append(
            VectorRecord(
                chunk_id=cid,
                vector=vec,
                document_id=doc_id,
                metadata={"idx": i},
            )
        )

    faiss_store.add(records)
    chroma_store.add(records)

    assert faiss_store.count() == num_records
    assert chroma_store.count() == num_records

    # 50 random queries: both must return the exact same top-1 hit
    for _ in range(num_queries):
        q = _random_normalized_vector(rng, dim)
        faiss_hits = faiss_store.search(q, k=1)
        chroma_hits = chroma_store.search(q, k=1)

        assert len(faiss_hits) == 1
        assert len(chroma_hits) == 1
        assert faiss_hits[0].chunk_id == chroma_hits[0].chunk_id
        # Scores should also be virtually identical (within 1e-4)
        assert math.isclose(faiss_hits[0].score, chroma_hits[0].score, abs_tol=1e-4)
