"""Document extractors implementing the Extractor protocol."""

from __future__ import annotations

from rag.ingestion.extractors.docx import DocxExtractor
from rag.ingestion.extractors.html import HtmlExtractor
from rag.ingestion.extractors.markdown import MarkdownExtractor
from rag.ingestion.extractors.pdf import PdfExtractor
from rag.ingestion.extractors.text import TextExtractor

__all__ = [
    "DocxExtractor",
    "HtmlExtractor",
    "MarkdownExtractor",
    "PdfExtractor",
    "TextExtractor",
]
