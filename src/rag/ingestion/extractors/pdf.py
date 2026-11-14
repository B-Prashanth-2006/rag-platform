"""PDF document extractor using PyMuPDF (fitz) with OCR fallback."""

from __future__ import annotations

import io
import os
from collections import Counter
from typing import Any

import pymupdf as fitz
import pytesseract
import structlog
from PIL import Image

from rag.domain.entities import ExtractedDocument, ExtractedPage
from rag.domain.errors import ExtractionError

logger = structlog.get_logger(__name__)


class PdfExtractor:
    """Extracts structured text, headings, and tables from PDF documents."""

    def supports(self, filename: str) -> bool:
        _, ext = os.path.splitext(filename)
        return ext.lower() == ".pdf"

    def extract(self, data: bytes, filename: str) -> ExtractedDocument:
        if not data:
            raise ExtractionError(f"Cannot extract from empty file: {filename}")

        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise ExtractionError(f"Failed to parse PDF {filename}: {exc}") from exc

        try:
            pages: list[ExtractedPage] = []
            for page_index in range(len(doc)):
                page = doc[page_index]
                page_num = page_index + 1

                # 1. Extract raw page text
                page_text = page.get_text()

                # 2. Extract tables
                tables: list[str] = []
                try:
                    tabs = page.find_tables()
                    for tab in tabs:
                        extracted_table = tab.extract()
                        if extracted_table:
                            formatted_rows = [
                                " | ".join(str(cell or "").strip() for cell in row)
                                for row in extracted_table
                            ]
                            tables.append("\n".join(formatted_rows))
                except Exception as exc:
                    logger.debug("Table detection failed on page", page=page_num, error=str(exc))

                # 3. Detect headings by font size relative to body
                headings = self._extract_headings(page)

                # 4. OCR fallback if page has under 20 extractable characters
                if len(page_text.strip()) < 20:
                    page_text = self._ocr_fallback(page, page_text, page_num)

                pages.append(
                    ExtractedPage(
                        number=page_num,
                        text=page_text,
                        headings=headings,
                        tables=tables,
                    )
                )

            metadata: dict[str, Any] = {
                "filename": filename,
                "page_count": len(pages),
                "format": "pdf",
            }
            return ExtractedDocument(pages=pages, metadata=metadata)
        finally:
            doc.close()

    def _extract_headings(self, page: fitz.Page) -> list[str]:
        """Detect headings by comparing font size of text spans to the modal body font size."""
        page_dict = page.get_text("dict")
        spans: list[dict[str, Any]] = []

        for block in page_dict.get("blocks", []):
            if block.get("type") == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            spans.append(span)

        if not spans:
            return []

        # Calculate modal font size (body size) weighted by text length
        font_counts: Counter[float] = Counter()
        for span in spans:
            size = round(float(span.get("size", 10.0)), 1)
            font_counts[size] += len(span.get("text", ""))

        body_font_size = font_counts.most_common(1)[0][0]

        headings: list[str] = []
        heading_threshold = body_font_size * 1.15

        for block in page_dict.get("blocks", []):
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    line_text = "".join(
                        span.get("text", "") for span in line.get("spans", "")
                    ).strip()
                    if not line_text:
                        continue
                    # Check if average font size of spans in line exceeds threshold
                    line_spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    if line_spans:
                        avg_size = sum(s.get("size", 0.0) for s in line_spans) / len(line_spans)
                        if avg_size >= heading_threshold:
                            headings.append(line_text)

        return headings

    def _ocr_fallback(self, page: fitz.Page, fallback_text: str, page_num: int) -> str:
        """Rasterize page and run Tesseract OCR. Log warning on failure and never crash."""
        try:
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(img)
            return ocr_text if ocr_text.strip() else fallback_text
        except Exception as exc:
            logger.warning(
                "OCR fallback failed or Tesseract is not installed",
                page=page_num,
                error=str(exc),
            )
            return fallback_text
