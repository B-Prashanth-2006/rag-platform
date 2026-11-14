# Module 2: Document Extraction and Ingestion

## Objective
Turn uploaded file bytes into clean, structured text with page metadata, and make
ingestion idempotent. No chunking or embedding yet (Modules 3 and 4).

## Scope (may touch)
src/rag/domain/entities.py (add new dataclasses only), src/rag/domain/interfaces.py
(add the Extractor Protocol only, do not modify existing ones), src/rag/domain/errors.py
(new errors only), src/rag/ingestion/, src/rag/config.py (max_upload_mb only),
pyproject.toml (listed deps only), tests/unit/ingestion/, tests/integration/,
Makefile, Dockerfile (tesseract install only)

## Requirements
1. **New domain types** (plain dataclasses, no third-party imports):
   - ExtractedPage(number: int starting at 1, text: str, headings: list[str], tables: list[str])
   - ExtractedDocument(pages: list[ExtractedPage], metadata: dict[str, Any])
   - Extractor Protocol: `supports(filename: str) -> bool` and
     `extract(data: bytes, filename: str) -> ExtractedDocument`
2. **Extractors** (ingestion/extractors/), one class each, all implementing Extractor:
   - PdfExtractor (PyMuPDF): per-page text, headings detected by font size relative to
     the body, tables as plain text rows.
   - DocxExtractor (python-docx): headings from Heading styles, tables; if the format has
     no pages, everything is page 1.
   - HtmlExtractor (BeautifulSoup): strip script/style/nav, keep h1-h6 as headings,
     tables as text.
   - MarkdownExtractor and TextExtractor: headings from `#` lines (markdown), UTF-8 with
     fallback decoding.
   - OCR fallback: a PDF page with almost no extractable text (under 20 characters) is
     rasterized and run through Tesseract (pytesseract). If Tesseract is not installed,
     log a warning and keep the empty page, never crash.
3. **ExtractorRegistry**: picks an extractor by filename extension; an unknown type
   raises UnsupportedFileTypeError.
4. **Normalization** (ingestion/cleaning.py), pure functions:
   - collapse repeated whitespace and blank lines, normalize unicode (NFKC), strip
     control characters
   - rejoin words hyphenated across line breaks ("exam-\nple" -> "example"), but keep
     real hyphens ("state-of-the-art")
   - remove repeated headers/footers: a line that appears at the top or bottom of at
     least 50% of pages (only for documents with 3 or more pages)
   - never change page numbers or reorder pages
5. **Content hash**: SHA-256 hex of the raw file bytes (not the text).
6. **IngestionService** (ingestion/service.py), constructed with DocumentRepository,
   JobRepository and ExtractorRegistry (dependency injection, no ORM imports):
   - `ingest(filename, data) -> IngestionResult(document, job, duplicate: bool,
     extracted: ExtractedDocument | None)`
   - If a document with the same hash exists and is not FAILED: do nothing and return it
     with duplicate=True.
   - If the existing one is FAILED: reuse it and reprocess.
   - Otherwise create a Document (PROCESSING) and a queued job, mark the job running,
     extract, clean, then mark the job succeeded and the document READY.
   - Extraction errors: mark the job failed with the error text and the document FAILED,
     then return the result (do not raise). Corrupt or empty files must not crash the
     service.
   - Use timezone-aware UTC timestamps. The service never commits (the caller owns the
     transaction).
7. **Limits**: reject files over a configurable max size (Settings.max_upload_mb,
   default 50) with FileTooLargeError.

## Acceptance criteria
- Fixture files are generated in code (PyMuPDF can create PDFs, python-docx can create
  DOCX), not committed as binaries. Include: multi-page PDF with headers and footers,
  PDF with a table, scanned (image-only) PDF, DOCX with headings and a table, HTML with
  scripts and nav, Markdown, TXT with odd encoding, an empty file, and a corrupt PDF.
- Tests verify page numbers, headings, header/footer removal, dehyphenation, and that
  re-uploading the same bytes returns duplicate=True with no new rows.
- The OCR test is skipped when Tesseract is not installed (pytest skipif), and passes
  locally with it.
- Property-style tests for cleaning: output never contains double spaces or control
  characters, and cleaning is idempotent (cleaning twice equals cleaning once).
- Run this exact sequence and it passes: uv run ruff check . && uv run ruff format
  --check . && uv run mypy src tests && uv run pytest
- No ORM imports in src/rag/ingestion/.

## Constraints
- Allowed new dependencies: pymupdf, python-docx, beautifulsoup4, pytesseract, pillow.
  Nothing else.
- Do not change existing Protocols, entities, models or repositories.
- Write tests first, then implementation.
- Finish with a summary of design decisions and anything unfinished.