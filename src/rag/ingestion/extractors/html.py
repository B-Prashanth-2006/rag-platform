"""HTML document extractor using BeautifulSoup."""

from __future__ import annotations

import os
from typing import Any

from bs4 import BeautifulSoup

from rag.domain.entities import ExtractedDocument, ExtractedPage


class HtmlExtractor:
    """Extracts text, headings, and tables from HTML documents into a single page."""

    def supports(self, filename: str) -> bool:
        _, ext = os.path.splitext(filename)
        return ext.lower() in (".html", ".htm")

    def extract(self, data: bytes, filename: str) -> ExtractedDocument:
        if not data:
            return ExtractedDocument(
                pages=[ExtractedPage(number=1, text="", headings=[], tables=[])],
                metadata={"filename": filename, "format": "html", "page_count": 1},
            )

        soup = BeautifulSoup(data, "html.parser")

        # Strip scripts, styles, navigation, and noscript elements
        for element in soup(["script", "style", "nav", "noscript"]):
            element.decompose()

        # Headings (h1 through h6)
        headings = [
            h.get_text(strip=True)
            for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
            if h.get_text(strip=True)
        ]

        # Tables
        tables: list[str] = []
        for table in soup.find_all("table"):
            table_rows: list[str] = []
            for tr in table.find_all("tr"):
                cells = [cell.get_text(strip=True) for cell in tr.find_all(["th", "td"])]
                if cells:
                    table_rows.append(" | ".join(cells))
            if table_rows:
                tables.append("\n".join(table_rows))

        # Main text
        text = soup.get_text(separator="\n", strip=True)

        page = ExtractedPage(
            number=1,
            text=text,
            headings=headings,
            tables=tables,
        )

        metadata: dict[str, Any] = {
            "filename": filename,
            "format": "html",
            "page_count": 1,
        }
        return ExtractedDocument(pages=[page], metadata=metadata)
