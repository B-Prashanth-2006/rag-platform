"""DOCX document extractor using python-docx."""

from __future__ import annotations

import io
import os
from typing import Any

from docx import Document as DocxDocument

from rag.domain.entities import ExtractedDocument, ExtractedPage
from rag.domain.errors import ExtractionError


class DocxExtractor:
    """Extracts text, headings, and tables from DOCX documents into a single page."""

    def supports(self, filename: str) -> bool:
        _, ext = os.path.splitext(filename)
        return ext.lower() == ".docx"

    def extract(self, data: bytes, filename: str) -> ExtractedDocument:
        if not data:
            raise ExtractionError(f"Cannot extract from empty file: {filename}")

        try:
            doc = DocxDocument(io.BytesIO(data))
        except Exception as exc:
            raise ExtractionError(f"Failed to parse DOCX {filename}: {exc}") from exc

        headings: list[str] = []
        body_parts: list[str] = []

        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            if p.style and p.style.name and p.style.name.startswith("Heading"):
                headings.append(text)
            body_parts.append(text)

        tables: list[str] = []
        for table in doc.tables:
            table_rows: list[str] = []
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells]
                table_rows.append(" | ".join(row_cells))
            if table_rows:
                tables.append("\n".join(table_rows))

        page_text = "\n\n".join(body_parts)

        page = ExtractedPage(
            number=1,
            text=page_text,
            headings=headings,
            tables=tables,
        )

        metadata: dict[str, Any] = {
            "filename": filename,
            "format": "docx",
            "page_count": 1,
        }
        return ExtractedDocument(pages=[page], metadata=metadata)
