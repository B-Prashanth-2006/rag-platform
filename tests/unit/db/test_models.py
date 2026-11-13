"""Unit tests for ORM model definitions — no live DB required."""

from sqlalchemy import inspect
from sqlalchemy.orm import DeclarativeBase

from rag.db.base import Base
from rag.db.models import ChunkOrm, DocumentOrm, IngestionJobOrm, QueryLogOrm


def test_base_is_declarative_base() -> None:
    assert issubclass(Base, DeclarativeBase)


def test_all_models_registered_in_base_metadata() -> None:
    table_names = set(Base.metadata.tables.keys())
    assert table_names == {"documents", "chunks", "ingestion_jobs", "query_logs"}


# ── DocumentOrm ──────────────────────────────────────────────────────────────


def test_document_orm_tablename() -> None:
    assert DocumentOrm.__tablename__ == "documents"


def test_document_orm_primary_key() -> None:
    mapper = inspect(DocumentOrm)
    pk_cols = [c.name for c in mapper.primary_key]
    assert pk_cols == ["id"]


def test_document_orm_columns_present() -> None:
    mapper = inspect(DocumentOrm)
    col_names = {c.key for c in mapper.column_attrs}
    assert col_names >= {"id", "filename", "content_hash", "status", "doc_metadata", "created_at", "updated_at"}


def test_document_orm_content_hash_unique() -> None:
    table = Base.metadata.tables["documents"]
    unique_cols = {
        col.name
        for uc in table.constraints
        if hasattr(uc, "columns")
        for col in uc.columns
        if hasattr(uc, "_type") or type(uc).__name__ == "UniqueConstraint"
    }
    # Check via the column's unique flag
    content_hash_col = table.c["content_hash"]
    assert content_hash_col.unique


# ── ChunkOrm ─────────────────────────────────────────────────────────────────


def test_chunk_orm_tablename() -> None:
    assert ChunkOrm.__tablename__ == "chunks"


def test_chunk_has_document_foreign_key() -> None:
    table = Base.metadata.tables["chunks"]
    fk_cols = {fk.parent.name for fk in table.foreign_keys}
    assert "document_id" in fk_cols


def test_chunk_orm_columns_present() -> None:
    mapper = inspect(ChunkOrm)
    col_names = {c.key for c in mapper.column_attrs}
    assert col_names >= {
        "id", "document_id", "index", "text", "page",
        "section", "token_count", "embedding_model_version",
    }


# ── IngestionJobOrm ───────────────────────────────────────────────────────────


def test_ingestion_job_orm_tablename() -> None:
    assert IngestionJobOrm.__tablename__ == "ingestion_jobs"


def test_ingestion_job_has_document_foreign_key() -> None:
    table = Base.metadata.tables["ingestion_jobs"]
    fk_cols = {fk.parent.name for fk in table.foreign_keys}
    assert "document_id" in fk_cols


# ── QueryLogOrm ───────────────────────────────────────────────────────────────


def test_query_log_orm_tablename() -> None:
    assert QueryLogOrm.__tablename__ == "query_logs"


def test_query_log_retrieved_chunk_ids_is_array() -> None:
    from sqlalchemy.dialects.postgresql import ARRAY

    table = Base.metadata.tables["query_logs"]
    col = table.c["retrieved_chunk_ids"]
    assert isinstance(col.type, ARRAY)

