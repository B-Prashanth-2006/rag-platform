"""Document normalization and cleaning functions.

Pure functions only: no ORM imports, no external service dependencies.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter

from rag.domain.entities import ExtractedDocument, ExtractedPage


def compute_content_hash(data: bytes) -> str:
    """Return the SHA-256 hexadecimal digest of raw file bytes."""
    return hashlib.sha256(data).hexdigest()


def normalize_text(text: str) -> str:
    """Apply Unicode NFKC normalization."""
    return unicodedata.normalize("NFKC", text)


def strip_control_characters(text: str) -> str:
    """Strip Unicode control characters, keeping newlines (\\n) and tabs (\\t)."""
    return "".join(
        ch for ch in text if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
    )


def rejoin_hyphenated_words(text: str) -> str:
    """Rejoin words broken across line breaks with a hyphen.

    Examples:
        "exam-\\nple" -> "example"
        "state-of-the-art" -> "state-of-the-art" (preserved because on same line)
    """
    # Matches a word ending with a hyphen immediately followed by a newline and optional indentation
    return re.sub(r"(\b[^\W\d_]+)-[ \t]*\r?\n[ \t]*([^\W\d_]+\b)", r"\1\2", text)


def collapse_whitespace(text: str) -> str:
    """Collapse repeated spaces within lines and repeated blank lines.

    Guarantees no consecutive double spaces exist in the output.
    """
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        # Collapse multiple horizontal whitespaces into a single space
        line_clean = re.sub(r"[^\S\n\r]+", " ", line).strip(" ")
        cleaned_lines.append(line_clean)

    joined = "\n".join(cleaned_lines)
    # Collapse 3 or more newlines into 2 (one blank line between paragraphs)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    # Guarantee no double spaces anywhere
    return re.sub(r" {2,}", " ", joined).strip()


def clean_text(text: str) -> str:
    """Clean a text string through normalization, control-character stripping,

    dehyphenation, and whitespace collapsing.
    """
    text = normalize_text(text)
    text = strip_control_characters(text)
    text = rejoin_hyphenated_words(text)
    return collapse_whitespace(text)


def remove_repeated_headers_footers(pages: list[ExtractedPage]) -> list[ExtractedPage]:
    """Remove lines appearing at the top or bottom of at least 50% of pages.

    Only applies to documents with 3 or more pages. Never reorders pages or alters
    page numbers.
    """
    if len(pages) < 3:
        return pages

    total_pages = len(pages)
    threshold = total_pages * 0.5

    top_lines: list[str] = []
    bottom_lines: list[str] = []

    for page in pages:
        lines = [line.strip() for line in page.text.split("\n") if line.strip()]
        if not lines:
            continue
        top_lines.append(lines[0])
        if len(lines) > 1:
            bottom_lines.append(lines[-1])
        else:
            # When there is only 1 line on the page, treat it only as a top candidate
            pass

    top_candidates = {line for line, count in Counter(top_lines).items() if count >= threshold}
    bottom_candidates = {
        line for line, count in Counter(bottom_lines).items() if count >= threshold
    }

    if not top_candidates and not bottom_candidates:
        return pages

    cleaned_pages: list[ExtractedPage] = []
    for page in pages:
        raw_lines = page.text.split("\n")
        first_non_empty: int | None = None
        last_non_empty: int | None = None

        for idx, line in enumerate(raw_lines):
            if line.strip():
                if first_non_empty is None:
                    first_non_empty = idx
                last_non_empty = idx

        lines_to_remove: set[int] = set()
        if first_non_empty is not None and raw_lines[first_non_empty].strip() in top_candidates:
            lines_to_remove.add(first_non_empty)

        if (
            last_non_empty is not None
            and last_non_empty not in lines_to_remove
            and raw_lines[last_non_empty].strip() in bottom_candidates
        ):
            lines_to_remove.add(last_non_empty)

        remaining_lines = [line for idx, line in enumerate(raw_lines) if idx not in lines_to_remove]
        new_text = "\n".join(remaining_lines).strip()
        cleaned_pages.append(
            ExtractedPage(
                number=page.number,
                text=new_text,
                headings=list(page.headings),
                tables=list(page.tables),
            )
        )

    return cleaned_pages


def clean_document(doc: ExtractedDocument) -> ExtractedDocument:
    """Normalize and clean an extracted document across all pages.

    Guaranteed to be idempotent: clean_document(clean_document(doc)) == clean_document(doc).
    Preserves page numbering and page order.
    """
    initial_pages = [
        ExtractedPage(
            number=page.number,
            text=clean_text(page.text),
            headings=[clean_text(h) for h in page.headings if clean_text(h)],
            tables=[clean_text(t) for t in page.tables if clean_text(t)],
        )
        for page in doc.pages
    ]

    header_footer_cleaned = remove_repeated_headers_footers(initial_pages)

    final_pages = [
        ExtractedPage(
            number=page.number,
            text=collapse_whitespace(page.text),
            headings=page.headings,
            tables=page.tables,
        )
        for page in header_footer_cleaned
    ]

    return ExtractedDocument(pages=final_pages, metadata=dict(doc.metadata))
