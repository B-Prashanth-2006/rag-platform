"""PostgreSQL implementation of ChunkLookup protocol.

The repository never commits or rolls back; the caller owns the transaction.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag.db.models import ChunkOrm
from rag.domain.entities import Chunk


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


class SqlChunkLookup:
    """ChunkLookup backed by a SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_many(self, chunk_ids: list[uuid.UUID]) -> list[Chunk]:
        """Fetch chunks in the exact requested order, silently skipping missing IDs."""
        if not chunk_ids:
            return []

        stmt = select(ChunkOrm).where(ChunkOrm.id.in_(chunk_ids))
        rows = self._session.scalars(stmt).all()
        chunk_map = {row.id: _to_entity(row) for row in rows}

        return [chunk_map[cid] for cid in chunk_ids if cid in chunk_map]
