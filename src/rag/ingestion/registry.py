"""Extractor registry for resolving extractors based on filename extension."""

from __future__ import annotations

from rag.domain.entities import ExtractedDocument
from rag.domain.errors import UnsupportedFileTypeError
from rag.domain.interfaces import Extractor
from rag.ingestion.extractors.docx import DocxExtractor
from rag.ingestion.extractors.html import HtmlExtractor
from rag.ingestion.extractors.markdown import MarkdownExtractor
from rag.ingestion.extractors.pdf import PdfExtractor
from rag.ingestion.extractors.text import TextExtractor


class ExtractorRegistry:
    """Registry that maps filenames to the appropriate Extractor implementation."""

    def __init__(self, extractors: list[Extractor] | None = None) -> None:
        if extractors is None:
            self._extractors: list[Extractor] = [
                PdfExtractor(),
                DocxExtractor(),
                HtmlExtractor(),
                MarkdownExtractor(),
                TextExtractor(),
            ]
        else:
            self._extractors = list(extractors)

    def register(self, extractor: Extractor) -> None:
        """Register a new extractor, prepending it to take precedence."""
        self._extractors.insert(0, extractor)

    def get(self, filename: str) -> Extractor:
        """Return the first extractor that supports filename, or raise UnsupportedFileTypeError."""
        for extractor in self._extractors:
            if extractor.supports(filename):
                return extractor
        raise UnsupportedFileTypeError(filename)

    def extract(self, data: bytes, filename: str) -> ExtractedDocument:
        """Resolve extractor and extract structured document."""
        extractor = self.get(filename)
        return extractor.extract(data, filename)
