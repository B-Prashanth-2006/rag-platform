"""SQLite-backed caching wrapper for embedders."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from rag.domain.entities import EmbeddingModelInfo
from rag.domain.interfaces import Embedder


class CachedEmbedder:
    """Wraps any Embedder with a persistent SQLite-based disk cache."""

    def __init__(
        self,
        embedder: Embedder,
        cache_path: Path | str = "data/embeddings_cache.db",
    ) -> None:
        self.embedder = embedder
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @property
    def info(self) -> EmbeddingModelInfo:
        """Return model metadata from wrapped embedder."""
        return self.embedder.info

    def _init_db(self) -> None:
        with self._lock, sqlite3.connect(self.cache_path, check_same_thread=False) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings_cache (
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    PRIMARY KEY (model_name, model_version, kind, text_hash)
                )
                """
            )
            conn.commit()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings with cache lookup, sending only misses in one batch."""
        if not texts:
            return []

        model_name = self.info.name
        model_version = self.info.version
        hashes = [self._hash_text(t) for t in texts]

        cache_hits: dict[str, list[float]] = {}
        with self._lock, sqlite3.connect(self.cache_path, check_same_thread=False) as conn:
            cursor = conn.cursor()
            # Fetch existing entries for these hashes
            placeholders = ",".join("?" for _ in hashes)
            cursor.execute(
                f"""
                SELECT text_hash, embedding
                FROM embeddings_cache
                WHERE model_name = ? AND model_version = ? AND kind = 'passage'
                  AND text_hash IN ({placeholders})
                """,
                [model_name, model_version, *hashes],
            )
            for row in cursor.fetchall():
                cache_hits[row[0]] = json.loads(row[1])

        # Identify misses (unique texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        missing_hashes: list[str] = []
        seen_miss_hashes: set[str] = set()

        for idx, (t, h) in enumerate(zip(texts, hashes, strict=True)):
            if h not in cache_hits and h not in seen_miss_hashes:
                missing_indices.append(idx)
                missing_texts.append(t)
                missing_hashes.append(h)
                seen_miss_hashes.add(h)

        if missing_texts:
            # Batch compute misses
            computed = self.embedder.embed_passages(missing_texts)
            rows_to_save: list[tuple[str, str, str, str, str]] = []
            for h, vec in zip(missing_hashes, computed, strict=True):
                cache_hits[h] = vec
                rows_to_save.append((model_name, model_version, "passage", h, json.dumps(vec)))

            with self._lock, sqlite3.connect(self.cache_path, check_same_thread=False) as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO embeddings_cache
                    (model_name, model_version, kind, text_hash, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows_to_save,
                )
                conn.commit()

        # Reconstruct in original order
        return [cache_hits[h] for h in hashes]

    def embed_query(self, text: str) -> list[float]:
        """Generate query embedding with cache lookup."""
        model_name = self.info.name
        model_version = self.info.version
        text_hash = self._hash_text(text)

        with self._lock, sqlite3.connect(self.cache_path, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT embedding
                FROM embeddings_cache
                WHERE model_name = ? AND model_version = ? AND kind = 'query' AND text_hash = ?
                """,
                (model_name, model_version, text_hash),
            )
            row = cursor.fetchone()
            if row is not None:
                return json.loads(row[0])  # type: ignore[no-any-return]

        # Compute miss
        vector = self.embedder.embed_query(text)
        with self._lock, sqlite3.connect(self.cache_path, check_same_thread=False) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings_cache
                (model_name, model_version, kind, text_hash, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                (model_name, model_version, "query", text_hash, json.dumps(vector)),
            )
            conn.commit()

        return vector
