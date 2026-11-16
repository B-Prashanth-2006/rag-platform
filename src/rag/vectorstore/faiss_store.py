"""FAISS-backed vector store supporting Flat and HNSW indexes with inner product."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from rag.domain.entities import SearchHit, VectorRecord
from rag.domain.errors import DimensionMismatchError


class FaissVectorStore:
    """Vector store implemented with FAISS (faiss-cpu) supporting Flat and HNSW indexes."""

    def __init__(
        self,
        dimension: int = 384,
        index_dir: Path | str = "data/indexes/faiss",
        index_type: str = "flat",
        m: int = 32,
        ef_search: int = 64,
        ef_construction: int = 64,
    ) -> None:
        self.dimension = dimension
        self.index_dir = Path(index_dir)
        self.index_type = index_type.lower()
        self.m = m
        self.ef_search = ef_search
        self.ef_construction = ef_construction

        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._index: faiss.IndexIDMap2
        self._uuid_to_id: dict[uuid.UUID, int] = {}
        self._id_to_uuid: dict[int, uuid.UUID] = {}
        self._records: dict[uuid.UUID, VectorRecord] = {}
        self._tombstones: set[int] = set()
        self._next_id: int = 0

        index_file = self.index_dir / "index.faiss"
        meta_file = self.index_dir / "meta.json"

        if index_file.exists() and meta_file.exists():
            self._load(index_file, meta_file)
        else:
            self._init_index()

    def _init_index(self) -> None:
        base: faiss.Index
        if self.index_type == "flat":
            base = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "hnsw":
            hnsw_index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_INNER_PRODUCT)
            hnsw_index.hnsw.efSearch = self.ef_search
            hnsw_index.hnsw.efConstruction = self.ef_construction
            base = hnsw_index
        else:
            raise ValueError(f"Unknown FAISS index type: {self.index_type}")

        self._index = faiss.IndexIDMap2(base)

    def _load(self, index_file: Path, meta_file: Path) -> None:
        raw_index = faiss.read_index(str(index_file))
        if isinstance(raw_index, faiss.IndexIDMap2):
            self._index = raw_index
        else:
            self._index = faiss.IndexIDMap2(raw_index)

        with open(meta_file, encoding="utf-8") as f:
            data = json.load(f)

        self.dimension = data["dimension"]
        self.index_type = data["index_type"]
        self._next_id = data["next_id"]
        self._tombstones = set(data.get("tombstones", []))

        self._uuid_to_id = {}
        self._id_to_uuid = {}
        self._records = {}

        for rec_dict in data["records"]:
            cid = uuid.UUID(rec_dict["chunk_id"])
            did = uuid.UUID(rec_dict["document_id"])
            int_id = rec_dict["int_id"]
            vec = rec_dict["vector"]
            meta = rec_dict["metadata"]

            record = VectorRecord(chunk_id=cid, vector=vec, document_id=did, metadata=meta)
            self._records[cid] = record
            self._uuid_to_id[cid] = int_id
            self._id_to_uuid[int_id] = cid

    def add(self, records: list[VectorRecord]) -> None:
        """Insert or replace vector records (upsert semantics)."""
        if not records:
            return

        for r in records:
            if len(r.vector) != self.dimension:
                raise DimensionMismatchError(expected=self.dimension, got=len(r.vector))

        # Handle replacements: delete existing chunks first
        existing_cids = [r.chunk_id for r in records if r.chunk_id in self._uuid_to_id]
        if existing_cids:
            self.delete_chunks(existing_cids)

        vectors: list[list[float]] = []
        ids: list[int] = []

        for r in records:
            int_id = self._next_id
            self._next_id += 1

            self._records[r.chunk_id] = r
            self._uuid_to_id[r.chunk_id] = int_id
            self._id_to_uuid[int_id] = r.chunk_id

            vectors.append(r.vector)
            ids.append(int_id)

        vecs_np = np.array(vectors, dtype=np.float32)
        ids_np = np.array(ids, dtype=np.int64)
        self._index.add_with_ids(vecs_np, ids_np)

    def search(
        self,
        vector: list[float],
        k: int,
        filter: dict[str, str | int | float | bool] | None = None,
    ) -> list[SearchHit]:
        """Search nearest vectors, optionally filtered by metadata."""
        if len(vector) != self.dimension:
            raise DimensionMismatchError(expected=self.dimension, got=len(vector))

        if self.count() == 0 or k <= 0:
            return []

        total_stored = self._index.ntotal
        if total_stored == 0:
            return []

        query_np = np.array([vector], dtype=np.float32)

        # Over-fetch if tombstones or filters are active
        needs_filtering = bool(filter) or bool(self._tombstones)
        k_fetch = min(max(k * 4, 100), total_stored) if needs_filtering else min(k, total_stored)

        while True:
            distances, indices = self._index.search(query_np, k_fetch)
            collected: list[SearchHit] = []

            for score, int_id in zip(distances[0], indices[0], strict=True):
                if int_id == -1:
                    continue
                if int_id in self._tombstones:
                    continue
                cid = self._id_to_uuid.get(int(int_id))
                if cid is None:
                    continue
                record = self._records.get(cid)
                if record is None:
                    continue

                if filter:
                    match = True
                    for f_key, f_val in filter.items():
                        if f_key == "document_id":
                            if str(record.document_id) != str(f_val):
                                match = False
                                break
                        elif record.metadata.get(f_key) != f_val:
                            match = False
                            break
                    if not match:
                        continue

                collected.append(SearchHit(chunk_id=cid, score=float(score)))

            # If we got enough hits, or we've already searched the whole index, stop
            if len(collected) >= k or k_fetch >= total_stored:
                break

            # Otherwise expand fetch window
            k_fetch = min(k_fetch * 2, total_stored)

        # Sort descending by score, tie broken by chunk_id string
        collected.sort(key=lambda h: (-h.score, str(h.chunk_id)))
        return collected[:k]

    def delete_chunks(self, chunk_ids: list[uuid.UUID]) -> None:
        """Remove vectors matching the given chunk IDs."""
        ids_to_remove: list[int] = []

        for cid in chunk_ids:
            if cid in self._uuid_to_id:
                int_id = self._uuid_to_id.pop(cid)
                self._id_to_uuid.pop(int_id, None)
                self._records.pop(cid, None)
                ids_to_remove.append(int_id)

        if not ids_to_remove:
            return

        if self.index_type == "flat":
            self._index.remove_ids(faiss.IDSelectorArray(np.array(ids_to_remove, dtype=np.int64)))
        else:
            # HNSW: tombstone deleted IDs
            self._tombstones.update(ids_to_remove)

    def delete_document(self, document_id: uuid.UUID) -> None:
        """Remove all vectors associated with *document_id*."""
        doc_chunk_ids = [cid for cid, r in self._records.items() if r.document_id == document_id]
        self.delete_chunks(doc_chunk_ids)

    def count(self) -> int:
        """Return the number of active, live vectors in the store."""
        return len(self._records)

    def compact(self) -> None:
        """Rebuild index for HNSW to permanently eliminate tombstones and reclaim memory."""
        if self.index_type != "hnsw":
            return

        live_records = list(self._records.values())
        self._init_index()
        self._tombstones.clear()
        self._uuid_to_id.clear()
        self._id_to_uuid.clear()
        self._records.clear()
        self._next_id = 0

        self.add(live_records)

    def persist(self) -> None:
        """Atomically write index and metadata to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

        meta_data: dict[str, Any] = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "next_id": self._next_id,
            "tombstones": list(self._tombstones),
            "records": [
                {
                    "chunk_id": str(r.chunk_id),
                    "document_id": str(r.document_id),
                    "int_id": self._uuid_to_id[r.chunk_id],
                    "vector": r.vector,
                    "metadata": r.metadata,
                }
                for r in self._records.values()
            ],
        }

        unique_suffix = f"tmp_{uuid.uuid4().hex}"
        temp_index = self.index_dir / f"index.faiss.{unique_suffix}"
        temp_meta = self.index_dir / f"meta.json.{unique_suffix}"

        final_index = self.index_dir / "index.faiss"
        final_meta = self.index_dir / "meta.json"

        faiss.write_index(self._index, str(temp_index))
        with open(temp_meta, "w", encoding="utf-8") as f:
            json.dump(meta_data, f)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_index, final_index)
        os.replace(temp_meta, final_meta)
