"""In-memory BM25 sparse index implementation using rank-bm25."""

from __future__ import annotations

import re
import uuid
from typing import Any

from rank_bm25 import BM25Okapi

from rag.domain.entities import Chunk, RetrievalFilter

_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)


def _tokenize(text: str) -> list[str]:
    """Tokenize text to lowercase alphanumeric tokens, preserving dashes for identifiers."""
    raw_tokens = re.findall(r"[\w-]+", text.lower())
    filtered = [t for t in raw_tokens if t not in _STOPWORDS]
    return filtered if filtered else raw_tokens


class Bm25Index:
    """In-memory sparse keyword index implementing the SparseIndex protocol."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: dict[uuid.UUID, Chunk] = {}
        self._chunk_ids: list[uuid.UUID] = []
        self._bm25: Any = None

    def _rebuild_index(self) -> None:
        self._chunk_ids = list(self._chunks.keys())
        if not self._chunk_ids:
            self._bm25 = None
            return

        corpus = [_tokenize(self._chunks[cid].text) for cid in self._chunk_ids]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        for word in self._bm25.idf:
            if self._bm25.idf[word] <= 0:
                self._bm25.idf[word] = 0.25

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index and refresh the BM25 model."""
        if not chunks:
            return

        for chunk in chunks:
            self._chunks[chunk.id] = chunk

        self._rebuild_index()

    def search(
        self,
        query: str,
        k: int,
        filter: RetrievalFilter | None = None,
    ) -> list[tuple[uuid.UUID, float]]:
        """Search the index for matching chunks, applying optional filters."""
        if self._bm25 is None or not self._chunks or k <= 0:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        raw_scores: list[float] = self._bm25.get_scores(tokens)
        scored_candidates: list[tuple[uuid.UUID, float]] = []

        allowed_docs = set(filter.document_ids) if filter and filter.document_ids else None

        for cid, score in zip(self._chunk_ids, raw_scores, strict=True):
            if score <= 0.0:
                continue

            chunk = self._chunks[cid]
            if allowed_docs is not None and chunk.document_id not in allowed_docs:
                continue

            scored_candidates.append((cid, float(score)))

        # Sort descending by score, tie-breaking ascending by chunk_id
        scored_candidates.sort(key=lambda item: (-item[1], item[0]))
        return scored_candidates[:k]

    def delete_document(self, document_id: uuid.UUID) -> None:
        """Remove all chunks associated with document_id and rebuild the index."""
        to_delete = [cid for cid, chunk in self._chunks.items() if chunk.document_id == document_id]
        if not to_delete:
            return

        for cid in to_delete:
            self._chunks.pop(cid, None)

        self._rebuild_index()

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)
