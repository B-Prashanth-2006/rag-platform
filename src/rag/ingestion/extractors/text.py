"""Plain text document extractor."""

from __future__ import annotations

import os
from typing import Any

from rag.domain.entities import ExtractedDocument, ExtractedPage


class TextExtractor:
    """Extracts plain text documents with fallback character decoding into a single page."""

    def supports(self, filename: str) -> bool:
        _, ext = os.path.splitext(filename)
        return ext.lower() == ".txt"

    def extract(self, data: bytes, filename: str) -> ExtractedDocument:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="replace")

        page = ExtractedPage(
            number=1,
            text=text,
            headings=[],
            tables=[],
        )

        metadata: dict[str, Any] = {
            "filename": filename,
            "format": "text",
            "page_count": 1,
        }
        return ExtractedDocument(pages=[page], metadata=metadata)
