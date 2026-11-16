"""ChromaDB-backed vector store implementation."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import chromadb

from rag.domain.entities import SearchHit, VectorRecord
from rag.domain.errors import DimensionMismatchError


class ChromaVectorStore:
    """Vector store implemented with ChromaDB PersistentClient."""

    def __init__(
        self,
        dimension: int = 384,
        index_dir: Path | str = "data/indexes/chroma",
        collection_name: str = "rag_chunks",
    ) -> None:
        self.dimension = dimension
        self.index_dir = Path(index_dir)
        self.collection_name = collection_name

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.index_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, records: list[VectorRecord]) -> None:
        """Insert or replace vector records (upsert semantics)."""
        if not records:
            return

        for r in records:
            if len(r.vector) != self.dimension:
                raise DimensionMismatchError(expected=self.dimension, got=len(r.vector))

        ids = [str(r.chunk_id) for r in records]
        embeddings: Any = [r.vector for r in records]
        # In Chroma, metadata values must be flat primitives (str, int, float, bool)
        metadatas = [{**r.metadata, "document_id": str(r.document_id)} for r in records]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

    def search(
        self,
        vector: list[float],
        k: int,
        filter: dict[str, str | int | float | bool] | None = None,
    ) -> list[SearchHit]:
        """Search nearest vectors, optionally filtered by metadata."""
        if len(vector) != self.dimension:
            raise DimensionMismatchError(expected=self.dimension, got=len(vector))

        total_stored = self.count()
        if total_stored == 0 or k <= 0:
            return []

        where_clause: dict[str, Any] | None = None
        if filter:
            conditions = [
                {key: str(val) if key == "document_id" else val} for key, val in filter.items()
            ]
            where_clause = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        n_results = min(k, total_stored)
        query_embeddings: Any = [vector]
        query_result = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where_clause,
            include=["distances"],
        )

        hits: list[SearchHit] = []
        query_ids = query_result.get("ids")
        query_dists = query_result.get("distances")

        if query_ids and query_dists:
            res_ids = query_ids[0]
            res_dists = query_dists[0]
            for id_str, dist in zip(res_ids, res_dists, strict=True):
                # Cosine distance to cosine similarity: score = 1.0 - distance
                score = 1.0 - float(dist)
                hits.append(SearchHit(chunk_id=uuid.UUID(id_str), score=score))

        # Sort descending by score, tie broken by chunk_id string
        hits.sort(key=lambda h: (-h.score, str(h.chunk_id)))
        return hits[:k]

    def delete_chunks(self, chunk_ids: list[uuid.UUID]) -> None:
        """Remove vectors matching the given chunk IDs."""
        if not chunk_ids:
            return

        ids = [str(cid) for cid in chunk_ids]
        existing = set(self._collection.get(ids=ids)["ids"])
        if existing:
            self._collection.delete(ids=list(existing))

    def delete_document(self, document_id: uuid.UUID) -> None:
        """Remove all vectors associated with *document_id*."""
        self._collection.delete(where={"document_id": str(document_id)})

    def count(self) -> int:
        """Return the total number of live vectors in the store."""
        return int(self._collection.count())

    def persist(self) -> None:
        """Persist state to disk (Chroma PersistentClient auto-persists)."""
        pass
