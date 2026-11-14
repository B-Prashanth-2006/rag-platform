"""Tests for ExtractorRegistry."""

from __future__ import annotations

import pytest

from rag.domain.errors import UnsupportedFileTypeError
from rag.ingestion.extractors.docx import DocxExtractor
from rag.ingestion.extractors.html import HtmlExtractor
from rag.ingestion.extractors.markdown import MarkdownExtractor
from rag.ingestion.extractors.pdf import PdfExtractor
from rag.ingestion.extractors.text import TextExtractor
from rag.ingestion.registry import ExtractorRegistry


def test_registry_resolves_all_supported_types() -> None:
    registry = ExtractorRegistry()

    assert isinstance(registry.get("document.pdf"), PdfExtractor)
    assert isinstance(registry.get("document.PDF"), PdfExtractor)
    assert isinstance(registry.get("report.docx"), DocxExtractor)
    assert isinstance(registry.get("index.html"), HtmlExtractor)
    assert isinstance(registry.get("index.htm"), HtmlExtractor)
    assert isinstance(registry.get("README.md"), MarkdownExtractor)
    assert isinstance(registry.get("README.markdown"), MarkdownExtractor)
    assert isinstance(registry.get("notes.txt"), TextExtractor)


def test_registry_unsupported_file_type() -> None:
    registry = ExtractorRegistry()

    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        registry.get("binary.exe")
    assert "binary.exe" in str(exc_info.value)

    with pytest.raises(UnsupportedFileTypeError):
        registry.get("archive.tar.gz")


def test_registry_extract_dispatches() -> None:
    registry = ExtractorRegistry()
    doc = registry.extract(b"Plain text content", "sample.txt")
    assert len(doc.pages) == 1
    assert "Plain text content" in doc.pages[0].text


def test_registry_custom_extractor_and_register() -> None:
    from rag.domain.entities import ExtractedDocument, ExtractedPage

    class CustomExtractor:
        def supports(self, filename: str) -> bool:
            return filename.endswith(".custom")

        def extract(self, data: bytes, filename: str) -> ExtractedDocument:
            return ExtractedDocument(
                pages=[ExtractedPage(number=1, text="custom", headings=[], tables=[])],
                metadata={"type": "custom"},
            )

    custom = CustomExtractor()
    registry = ExtractorRegistry(extractors=[custom])
    assert registry.get("test.custom") is custom

    # Test register() method
    empty_registry = ExtractorRegistry(extractors=[])
    empty_registry.register(custom)
    assert empty_registry.get("another.custom") is custom
