"""SQLAlchemy 2.0 ORM models.

Each model mirrors the domain entity of the same name but lives only in ``db/``.
Domain code never imports from here; only repositories (also in ``db/``) do.

Column naming note:
  - ``DocumentOrm.doc_metadata`` maps to the DB column ``metadata`` to avoid
    shadowing ``DeclarativeBase.metadata``.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag.db.base import Base
from rag.db.types import DOCUMENT_STATUS_ENUM, JOB_STATUS_ENUM
from rag.domain.entities import DocumentStatus, JobStatus


class DocumentOrm(Base):
    """ORM model for the ``documents`` table."""

    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("content_hash", name="uq_documents_content_hash"),)

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    status: Mapped[DocumentStatus] = mapped_column(
        DOCUMENT_STATUS_ENUM, nullable=False, default=DocumentStatus.PENDING
    )
    doc_metadata: Mapped[dict[str, object]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships (back-populated by child models)
    chunks: Mapped[list[ChunkOrm]] = relationship(
        "ChunkOrm", back_populates="document", cascade="all, delete-orphan"
    )
    ingestion_jobs: Mapped[list[IngestionJobOrm]] = relationship(
        "IngestionJobOrm", back_populates="document", cascade="all, delete-orphan"
    )


class ChunkOrm(Base):
    """ORM model for the ``chunks`` table."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_model_version: Mapped[str | None] = mapped_column(Text, nullable=True)

    document: Mapped[DocumentOrm] = relationship("DocumentOrm", back_populates="chunks")


class IngestionJobOrm(Base):
    """ORM model for the ``ingestion_jobs`` table."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[JobStatus] = mapped_column(
        JOB_STATUS_ENUM, nullable=False, default=JobStatus.QUEUED
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    document: Mapped[DocumentOrm] = relationship(
        "DocumentOrm", back_populates="ingestion_jobs"
    )


class QueryLogOrm(Base):
    """ORM model for the ``query_logs`` table."""

    __tablename__ = "query_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunk_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)), nullable=False, default=list
    )
    latency_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

