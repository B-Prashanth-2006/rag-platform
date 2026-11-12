"""Domain entities for the RAG platform.

Plain dataclasses only — no SQLAlchemy imports, no external dependencies.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
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
