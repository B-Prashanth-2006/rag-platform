"""PostgreSQL implementation of ChunkRepository.

The repository never commits. The caller owns the transaction.
begin_nested() (SAVEPOINT) is used in add_many so a constraint violation
does not discard other pending work in the outer session.
"""

from __future__ import annotations

import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from rag.db.models import ChunkOrm
from rag.domain.entities import Chunk
from rag.domain.errors import DuplicateChunkIndexError


def _to_entity(row: ChunkOrm) -> Chunk:
    return Chunk(
        id=row.id,
        document_id=row.document_id,
        index=row.index,
        text=row.text,
        page=row.page,
        section=row.section,
        token_count=row.token_count,
        embedding_model_version=row.embedding_model_version,
    )


def _to_orm(chunk: Chunk) -> ChunkOrm:
    return ChunkOrm(
        id=chunk.id,
        document_id=chunk.document_id,
        index=chunk.index,
        text=chunk.text,
        page=chunk.page,
        section=chunk.section,
        token_count=chunk.token_count,
        embedding_model_version=chunk.embedding_model_version,
    )


class SqlChunkRepository:
    """ChunkRepository backed by a SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_many(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        # Determine which document_id to report in a potential error; all chunks
        # in one add_many call must belong to the same document (spec invariant).
        document_id = chunks[0].document_id
        try:
            with self._session.begin_nested():  # SAVEPOINT
                self._session.add_all([_to_orm(c) for c in chunks])
        except IntegrityError as exc:
            orig = str(exc.orig)
            if "uq_chunks_document_id_index" in orig or ("document_id" in orig and "index" in orig):
                raise DuplicateChunkIndexError(document_id, chunks[0].index) from exc
            raise

    def list_by_document(self, document_id: uuid.UUID) -> list[Chunk]:
        stmt = select(ChunkOrm).where(ChunkOrm.document_id == document_id).order_by(ChunkOrm.index)
        return [_to_entity(row) for row in self._session.scalars(stmt).all()]

    def delete_by_document(self, document_id: uuid.UUID) -> None:
        stmt = delete(ChunkOrm).where(ChunkOrm.document_id == document_id)
        self._session.execute(stmt)
        self._session.flush()

    def count_by_model_version(self, version: str) -> int:
        stmt = (
            select(func.count())
            .select_from(ChunkOrm)
            .where(ChunkOrm.embedding_model_version == version)
        )
        result = self._session.scalar(stmt)
        return int(result) if result is not None else 0
