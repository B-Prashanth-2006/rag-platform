"""Domain entities for the RAG platform.

Plain dataclasses only — no SQLAlchemy imports, no external dependencies.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class DocumentStatus(StrEnum):
    """Lifecycle states of an ingested document."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class JobStatus(StrEnum):
    """Lifecycle states of an ingestion job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Document:
    """Corresponds to the ``documents`` table."""

    id: uuid.UUID
    filename: str
    content_hash: str  # SHA-256; unique across documents (idempotent ingestion)
    status: DocumentStatus
    metadata: dict[str, Any]  # arbitrary JSONB bag
    created_at: datetime
    updated_at: datetime


@dataclass
class Chunk:
    """Corresponds to the ``chunks`` table.

    Invariants (enforced in ``__post_init__``):
    - ``index`` must be >= 0
    - ``token_count`` must be > 0
    """

    id: uuid.UUID
    document_id: uuid.UUID
    index: int  # zero-based position within the parent document
    text: str
    page: int | None
    section: str | None
    token_count: int
    embedding_model_version: str | None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must be >= 0, got {self.index}")
        if self.token_count <= 0:
            raise ValueError(f"token_count must be > 0, got {self.token_count}")


@dataclass
class IngestionJob:
    """Corresponds to the ``ingestion_jobs`` table."""

    id: uuid.UUID
    document_id: uuid.UUID
    status: JobStatus  # e.g. QUEUED | RUNNING | SUCCEEDED | FAILED
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None


@dataclass
class QueryLog:
    """Corresponds to the ``query_logs`` table."""

    id: uuid.UUID
    query: str
    retrieved_chunk_ids: list[uuid.UUID]
    latency_ms: int
    answer: str | None
    feedback: str | None
    created_at: datetime


@dataclass
class ExtractedPage:
    """A single extracted page from a document.

    Invariants:
    - ``number`` must be >= 1
    """

    number: int
    text: str
    headings: list[str]
    tables: list[str]

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError(f"number must be >= 1, got {self.number}")


@dataclass
class ExtractedDocument:
    """In-memory representation of an extracted document across its pages."""

    pages: list[ExtractedPage]
    metadata: dict[str, Any]


@dataclass
class ChunkDraft:
    """In-memory draft of a chunk before persistence.

    Has no database IDs or timestamps; those are assigned when persisting.

    Invariants (enforced in ``__post_init__``):
    - ``index`` must be >= 0
    - ``token_count`` must be > 0
    """

    index: int
    text: str
    page: int | None
    section: str | None
    token_count: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must be >= 0, got {self.index}")
        if self.token_count <= 0:
            raise ValueError(f"token_count must be > 0, got {self.token_count}")


@dataclass
class EmbeddingModelInfo:
    """Metadata identifying the embedding model and its characteristics."""

    name: str
    version: str
    dimension: int
    max_tokens: int


@dataclass
class VectorRecord:
    """A vector embedding with its chunk and document metadata."""

    chunk_id: uuid.UUID
    vector: list[float]
    document_id: uuid.UUID
    metadata: dict[str, str | int | float | bool]


@dataclass
class SearchHit:
    """A search result returned by a vector store.

    Scores represent cosine similarity in [-1, 1], where higher score = more similar.
    """

    chunk_id: uuid.UUID
    score: float


_VALID_RETRIEVAL_SOURCES = frozenset({"dense", "sparse", "fused", "reranked"})


@dataclass
class RetrievalFilter:
    """Filter applied to retrieval stages to restrict matching chunks."""

    document_ids: list[uuid.UUID] | None = None


@dataclass
class RetrievedChunk:
    """A chunk retrieved from a retrieval stage with score and source attribution."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    text: str
    page: int | None
    section: str | None
    score: float
    source: str

    def __post_init__(self) -> None:
        if self.source not in _VALID_RETRIEVAL_SOURCES:
            raise ValueError(
                f"Invalid source '{self.source}'. Must be one of {_VALID_RETRIEVAL_SOURCES}"
            )


@dataclass
class StageTrace:
    """Diagnostic trace of an individual retrieval stage."""

    stage: str
    latency_ms: float
    results: list[tuple[uuid.UUID, float]]


@dataclass
class RetrievalResult:
    """Combined output of a retrieval pipeline or stage."""

    chunks: list[RetrievedChunk]
    trace: list[StageTrace] = field(default_factory=list)

    @property
    def top_score(self) -> float | None:
        """Return the highest score among retrieved chunks, or None if empty."""
        return self.chunks[0].score if self.chunks else None
