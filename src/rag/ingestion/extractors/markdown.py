"""Markdown document extractor."""

from __future__ import annotations

import os
import re
from typing import Any

from rag.domain.entities import ExtractedDocument, ExtractedPage


class MarkdownExtractor:
    """Extracts text and headings from Markdown documents into a single page."""

    def supports(self, filename: str) -> bool:
        _, ext = os.path.splitext(filename)
        return ext.lower() in (".md", ".markdown")

    def extract(self, data: bytes, filename: str) -> ExtractedDocument:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="replace")

        # Headings from `#` lines
        headings = re.findall(r"^#{1,6}\s+(.+)$", text, flags=re.MULTILINE)
        headings = [h.strip() for h in headings if h.strip()]

        # Tables (markdown style table blocks)
        tables: list[str] = []
        table_lines: list[str] = []
        for line in text.split("\n"):
            if "|" in line:
                table_lines.append(line.strip())
            else:
                if len(table_lines) >= 2:
                    tables.append("\n".join(table_lines))
                table_lines = []
        if len(table_lines) >= 2:
            tables.append("\n".join(table_lines))

        page = ExtractedPage(
            number=1,
            text=text,
            headings=headings,
            tables=tables,
        )

        metadata: dict[str, Any] = {
            "filename": filename,
            "format": "markdown",
            "page_count": 1,
        }
        return ExtractedDocument(pages=[page], metadata=metadata)
