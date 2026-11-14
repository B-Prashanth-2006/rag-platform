"""Unit tests for domain entities — written first (TDD red phase)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from rag.domain.entities import Chunk, Document, DocumentStatus, IngestionJob, JobStatus, QueryLog

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _make_document(**overrides: Any) -> Document:
    defaults: dict[str, Any] = {
        "id": uuid.uuid4(),
        "filename": "report.pdf",
        "content_hash": "abc123",
        "status": DocumentStatus.PENDING,
        "metadata": {},
        "created_at": _now(),
        "updated_at": _now(),
    }
    defaults.update(overrides)
    return Document(**defaults)


def _make_chunk(**overrides: Any) -> Chunk:
    defaults: dict[str, Any] = {
        "id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "index": 0,
        "text": "Hello world",
        "page": 1,
        "section": "Introduction",
        "token_count": 10,
        "embedding_model_version": "text-embedding-3-small-v1",
    }
    defaults.update(overrides)
    return Chunk(**defaults)


def _make_job(**overrides: Any) -> IngestionJob:
    defaults: dict[str, Any] = {
        "id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "status": JobStatus.QUEUED,
        "error": None,
        "started_at": None,
        "finished_at": None,
    }
    defaults.update(overrides)
    return IngestionJob(**defaults)


def _make_query_log(**overrides: Any) -> QueryLog:
    defaults: dict[str, Any] = {
        "id": uuid.uuid4(),
        "query": "What is RAG?",
        "retrieved_chunk_ids": [uuid.uuid4(), uuid.uuid4()],
        "latency_ms": 42,
        "answer": "Retrieval Augmented Generation",
        "feedback": None,
        "created_at": _now(),
    }
    defaults.update(overrides)
    return QueryLog(**defaults)


# ---------------------------------------------------------------------------
# DocumentStatus
# ---------------------------------------------------------------------------


class TestDocumentStatus:
    def test_all_variants_exist(self) -> None:
        assert DocumentStatus.PENDING
        assert DocumentStatus.PROCESSING
        assert DocumentStatus.READY
        assert DocumentStatus.FAILED

    def test_values_are_strings(self) -> None:
        for member in DocumentStatus:
            assert isinstance(member.value, str)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class TestDocument:
    def test_instantiation(self) -> None:
        doc = _make_document()
        assert doc.filename == "report.pdf"
        assert doc.status == DocumentStatus.PENDING

    def test_equality_by_fields(self) -> None:
        doc_id = uuid.uuid4()
        ts = _now()
        doc1 = _make_document(id=doc_id, created_at=ts, updated_at=ts)
        doc2 = _make_document(id=doc_id, created_at=ts, updated_at=ts)
        assert doc1 == doc2

    def test_metadata_is_dict(self) -> None:
        doc = _make_document(metadata={"pages": 10, "author": "Alice"})
        assert doc.metadata["pages"] == 10

    def test_all_status_values_accepted(self) -> None:
        for status in DocumentStatus:
            doc = _make_document(status=status)
            assert doc.status == status

    def test_id_is_uuid(self) -> None:
        doc = _make_document()
        assert isinstance(doc.id, uuid.UUID)

    def test_no_sqlalchemy_import(self) -> None:
        """entities.py must not contain SQLAlchemy import statements."""
        import re

        import rag.domain.entities as mod

        source_file = mod.__file__
        assert source_file is not None
        with open(source_file) as f:
            source = f.read()

        # Match actual import lines only, not comments or docstrings
        import_pattern = re.compile(
            r"^\s*(import sqlalchemy|from sqlalchemy)", re.MULTILINE | re.IGNORECASE
        )
        assert not import_pattern.search(source), "entities.py must not import SQLAlchemy"


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_instantiation(self) -> None:
        chunk = _make_chunk()
        assert chunk.text == "Hello world"
        assert chunk.index == 0

    def test_optional_fields_can_be_none(self) -> None:
        chunk = _make_chunk(page=None, section=None, embedding_model_version=None)
        assert chunk.page is None
        assert chunk.section is None
        assert chunk.embedding_model_version is None

    def test_index_cannot_be_negative(self) -> None:
        with pytest.raises(ValueError, match="index"):
            _make_chunk(index=-1)

    def test_token_count_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="token_count"):
            _make_chunk(token_count=0)

    def test_document_id_is_uuid(self) -> None:
        chunk = _make_chunk()
        assert isinstance(chunk.document_id, uuid.UUID)

    def test_equality_by_fields(self) -> None:
        cid = uuid.uuid4()
        did = uuid.uuid4()
        c1 = _make_chunk(id=cid, document_id=did, index=3)
        c2 = _make_chunk(id=cid, document_id=did, index=3)
        assert c1 == c2


# ---------------------------------------------------------------------------
# IngestionJob
# ---------------------------------------------------------------------------


class TestJobStatus:
    def test_all_variants_exist(self) -> None:
        assert JobStatus.QUEUED
        assert JobStatus.RUNNING
        assert JobStatus.SUCCEEDED
        assert JobStatus.FAILED

    def test_values_are_strings(self) -> None:
        for member in JobStatus:
            assert isinstance(member.value, str)

    def test_queued_value(self) -> None:
        assert JobStatus.QUEUED.value == "queued"

    def test_succeeded_value(self) -> None:
        assert JobStatus.SUCCEEDED.value == "succeeded"


class TestIngestionJob:
    def test_instantiation(self) -> None:
        job = _make_job()
        assert job.status == JobStatus.QUEUED
        assert job.error is None

    def test_all_status_values_accepted(self) -> None:
        for status in JobStatus:
            job = _make_job(status=status)
            assert job.status == status

    def test_failed_with_error(self) -> None:
        job = _make_job(status=JobStatus.FAILED, error="Timeout reading source file")
        assert job.status == JobStatus.FAILED
        assert job.error == "Timeout reading source file"

    def test_running_has_started_at(self) -> None:
        ts = _now()
        job = _make_job(status=JobStatus.RUNNING, started_at=ts)
        assert job.started_at == ts
        assert job.finished_at is None

    def test_succeeded_has_both_timestamps(self) -> None:
        ts = _now()
        job = _make_job(status=JobStatus.SUCCEEDED, started_at=ts, finished_at=ts)
        assert job.started_at == ts
        assert job.finished_at == ts

    def test_id_is_uuid(self) -> None:
        job = _make_job()
        assert isinstance(job.id, uuid.UUID)


# ---------------------------------------------------------------------------
# QueryLog
# ---------------------------------------------------------------------------


class TestQueryLog:
    def test_instantiation(self) -> None:
        log = _make_query_log()
        assert log.query == "What is RAG?"
        assert log.latency_ms == 42

    def test_retrieved_chunk_ids_are_uuids(self) -> None:
        log = _make_query_log()
        assert all(isinstance(cid, uuid.UUID) for cid in log.retrieved_chunk_ids)

    def test_optional_fields_can_be_none(self) -> None:
        log = _make_query_log(answer=None, feedback=None)
        assert log.answer is None
        assert log.feedback is None

    def test_empty_retrieved_chunks(self) -> None:
        log = _make_query_log(retrieved_chunk_ids=[])
        assert log.retrieved_chunk_ids == []

    def test_id_is_uuid(self) -> None:
        log = _make_query_log()
        assert isinstance(log.id, uuid.UUID)


# ---------------------------------------------------------------------------
# ExtractedPage and ExtractedDocument
# ---------------------------------------------------------------------------


class TestExtractedPage:
    def test_instantiation(self) -> None:
        from rag.domain.entities import ExtractedPage

        page = ExtractedPage(
            number=1,
            text="Sample page text",
            headings=["Heading 1"],
            tables=["Row 1 | Col 1"],
        )
        assert page.number == 1
        assert page.text == "Sample page text"
        assert page.headings == ["Heading 1"]
        assert page.tables == ["Row 1 | Col 1"]

    def test_number_must_be_positive(self) -> None:
        from rag.domain.entities import ExtractedPage

        with pytest.raises(ValueError, match="number must be >= 1"):
            ExtractedPage(number=0, text="", headings=[], tables=[])

        with pytest.raises(ValueError, match="number must be >= 1"):
            ExtractedPage(number=-1, text="", headings=[], tables=[])


class TestExtractedDocument:
    def test_instantiation(self) -> None:
        from rag.domain.entities import ExtractedDocument, ExtractedPage

        page1 = ExtractedPage(number=1, text="Page 1", headings=[], tables=[])
        page2 = ExtractedPage(number=2, text="Page 2", headings=[], tables=[])
        doc = ExtractedDocument(pages=[page1, page2], metadata={"author": "Test"})

        assert len(doc.pages) == 2
        assert doc.metadata == {"author": "Test"}


class TestIngestionErrors:
    def test_file_too_large_error(self) -> None:
        from rag.domain.errors import FileTooLargeError

        err = FileTooLargeError(size_bytes=60_000_000, max_bytes=52_428_800)
        assert err.size_bytes == 60_000_000
        assert err.max_bytes == 52_428_800
        assert "exceeds maximum allowed size" in str(err)

    def test_unsupported_file_type_error(self) -> None:
        from rag.domain.errors import UnsupportedFileTypeError

        err = UnsupportedFileTypeError("bad.xyz")
        assert err.filename == "bad.xyz"
        assert "bad.xyz" in str(err)

    def test_extraction_error(self) -> None:
        from rag.domain.errors import ExtractionError

        err = ExtractionError("Corrupt PDF file")
        assert err.message == "Corrupt PDF file"
        assert "Corrupt PDF file" in str(err)
