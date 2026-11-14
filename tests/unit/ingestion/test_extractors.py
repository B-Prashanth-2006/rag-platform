"""Tests for document extractors (PDF, DOCX, HTML, Markdown, Text) and OCR fallback."""

from __future__ import annotations

import shutil
from unittest.mock import patch

import pytest

from rag.domain.entities import ExtractedDocument
from rag.domain.errors import ExtractionError
from rag.domain.interfaces import Extractor
from rag.ingestion.extractors.docx import DocxExtractor
from rag.ingestion.extractors.html import HtmlExtractor
from rag.ingestion.extractors.markdown import MarkdownExtractor
from rag.ingestion.extractors.pdf import PdfExtractor
from rag.ingestion.extractors.text import TextExtractor

from .conftest import (
    generate_corrupt_pdf,
    generate_docx_with_headings_and_table,
    generate_empty_file,
    generate_html_with_scripts_and_nav,
    generate_markdown,
    generate_multipage_pdf_with_headers_footers,
    generate_pdf_with_table,
    generate_scanned_image_pdf,
    generate_txt_odd_encoding,
)


def test_pdf_extractor_supports() -> None:
    extractor = PdfExtractor()
    assert extractor.supports("file.pdf")
    assert extractor.supports("DOCUMENT.PDF")
    assert not extractor.supports("file.docx")
    assert not extractor.supports("file.txt")


def test_pdf_extractor_multipage() -> None:
    extractor = PdfExtractor()
    data = generate_multipage_pdf_with_headers_footers()
    doc = extractor.extract(data, "annual_report.pdf")

    assert isinstance(doc, ExtractedDocument)
    assert len(doc.pages) == 3
    assert [p.number for p in doc.pages] == [1, 2, 3]

    # Verify headings detected by font size
    assert any("Section 1: Overview" in h for h in doc.pages[0].headings)
    assert any("Section 2: Overview" in h for h in doc.pages[1].headings)
    assert any("Section 3: Overview" in h for h in doc.pages[2].headings)

    # Verify body text
    assert "ACME Corporation Annual Report" in doc.pages[0].text
    assert "Section 1: Overview" in doc.pages[0].text


def test_pdf_extractor_tables() -> None:
    extractor = PdfExtractor()
    data = generate_pdf_with_table()
    doc = extractor.extract(data, "table.pdf")

    assert len(doc.pages) == 1
    page = doc.pages[0]
    assert len(page.tables) >= 1
    table_text = " ".join(page.tables)
    assert "Quarter" in table_text or "Revenue" in table_text
    assert "Q1 2026" in table_text or "$1.2M" in table_text


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed on host")
def test_pdf_ocr_fallback_with_tesseract() -> None:
    extractor = PdfExtractor()
    data = generate_scanned_image_pdf()
    doc = extractor.extract(data, "scanned.pdf")

    assert len(doc.pages) == 1
    page = doc.pages[0]
    # OCR should have recognized text from the image
    assert "SCANNED" in page.text or "INVOICE" in page.text or "99482" in page.text


def test_pdf_ocr_fallback_when_tesseract_missing() -> None:
    extractor = PdfExtractor()
    data = generate_scanned_image_pdf()

    # Simulate tesseract missing / raising FileNotFoundError or TesseractNotFoundError
    with patch("pytesseract.image_to_string", side_effect=FileNotFoundError("tesseract not found")):
        doc = extractor.extract(data, "scanned.pdf")
        # Should not crash; returns page without failing
        assert len(doc.pages) == 1
        assert doc.pages[0].number == 1


def test_pdf_extractor_corrupt_file() -> None:
    extractor = PdfExtractor()
    corrupt_data = generate_corrupt_pdf()
    with pytest.raises(ExtractionError):
        extractor.extract(corrupt_data, "bad.pdf")


def test_pdf_extractor_empty_file() -> None:
    extractor = PdfExtractor()
    empty_data = generate_empty_file()
    with pytest.raises(ExtractionError):
        extractor.extract(empty_data, "empty.pdf")


def test_docx_extractor() -> None:
    extractor = DocxExtractor()
    assert extractor.supports("report.docx")
    assert extractor.supports("REPORT.DOCX")
    assert not extractor.supports("report.doc")

    data = generate_docx_with_headings_and_table()
    doc = extractor.extract(data, "test.docx")

    assert len(doc.pages) == 1
    page = doc.pages[0]
    assert page.number == 1
    assert "Docx Main Document Title" in page.headings
    assert "Subsection 1.1" in page.headings
    assert len(page.tables) >= 1
    table_text = " ".join(page.tables)
    assert "Col A" in table_text
    assert "Val 1" in table_text
    assert "This is an introductory paragraph" in page.text


def test_docx_extractor_empty_file() -> None:
    extractor = DocxExtractor()
    with pytest.raises(ExtractionError):
        extractor.extract(b"", "empty.docx")


def test_html_extractor() -> None:
    extractor = HtmlExtractor()
    assert extractor.supports("page.html")
    assert extractor.supports("page.htm")
    assert not extractor.supports("page.txt")

    data = generate_html_with_scripts_and_nav()
    doc = extractor.extract(data, "index.html")

    assert len(doc.pages) == 1
    page = doc.pages[0]
    assert page.number == 1

    # Script, style, nav must be stripped
    assert "malicious" not in page.text
    assert "background-color" not in page.text
    assert "Home" not in page.text
    assert "About" not in page.text

    # Headings extracted
    assert "Page Main Heading" in page.headings
    assert "Secondary Heading" in page.headings

    # Tables extracted
    assert len(page.tables) >= 1
    table_text = " ".join(page.tables)
    assert "Widget" in table_text
    assert "$10" in table_text

    # Content
    assert "Introductory paragraph in HTML." in page.text


def test_html_extractor_empty() -> None:
    extractor = HtmlExtractor()
    doc = extractor.extract(b"", "empty.html")
    assert len(doc.pages) == 1
    assert doc.pages[0].text == ""


def test_markdown_extractor() -> None:
    extractor = MarkdownExtractor()
    assert extractor.supports("README.md")
    assert extractor.supports("notes.markdown")
    assert not extractor.supports("notes.txt")

    data = generate_markdown()
    doc = extractor.extract(data, "README.md")

    assert len(doc.pages) == 1
    page = doc.pages[0]
    assert page.number == 1
    assert "Markdown Project Report" in page.headings
    assert "Methodology" in page.headings
    assert "Discussion" in page.headings
    assert "We evaluated several approaches." in page.text


def test_text_extractor() -> None:
    extractor = TextExtractor()
    assert extractor.supports("notes.txt")
    assert extractor.supports("NOTES.TXT")
    assert not extractor.supports("notes.md")

    data = generate_txt_odd_encoding()
    doc = extractor.extract(data, "french.txt")

    assert len(doc.pages) == 1
    page = doc.pages[0]
    assert page.number == 1
    assert page.headings == []
    assert page.tables == []
    assert "Rapport Financier" in page.text
    assert "café" in page.text or "caf" in page.text


def test_extractors_implement_protocol() -> None:
    extractors: list[Extractor] = [
        PdfExtractor(),
        DocxExtractor(),
        HtmlExtractor(),
        MarkdownExtractor(),
        TextExtractor(),
    ]
    for ext in extractors:
        assert isinstance(ext, Extractor)
