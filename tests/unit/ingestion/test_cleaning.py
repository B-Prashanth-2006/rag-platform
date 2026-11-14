"""Tests for document normalization, dehyphenation, header/footer removal, and cleaning."""

from __future__ import annotations

import unicodedata

import pytest

from rag.domain.entities import ExtractedDocument, ExtractedPage
from rag.ingestion.cleaning import (
    clean_document,
    clean_text,
    collapse_whitespace,
    compute_content_hash,
    normalize_text,
    rejoin_hyphenated_words,
    remove_repeated_headers_footers,
    strip_control_characters,
)


def test_compute_content_hash() -> None:
    data = b"hello world\n"
    expected = "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
    assert compute_content_hash(data) == expected


def test_normalize_unicode_nfkc() -> None:
    # Full-width characters and ligatures
    text = "ﬁle ｗｏｒｌｄ"
    normalized = normalize_text(text)
    assert normalized == "file world"


def test_strip_control_characters() -> None:
    # ASCII control chars except newline and tab
    text = "Hello\x00\x07\x1bWorld\nLine 2\tTabbed"
    cleaned = strip_control_characters(text)
    assert cleaned == "HelloWorld\nLine 2\tTabbed"


def test_rejoin_hyphenated_words() -> None:
    text = "This is an exam-\nple of dehyphen-\n  ation.\nHowever, state-of-the-art remains intact."
    rejoined = rejoin_hyphenated_words(text)
    assert "example" in rejoined
    assert "dehyphenation" in rejoined
    assert "state-of-the-art" in rejoined


def test_rejoin_hyphenated_words_crlf() -> None:
    text = "Multi-\r\nline break should also be dehyphen-\r\nated."
    rejoined = rejoin_hyphenated_words(text)
    assert "Multiline" in rejoined
    assert "dehyphenated" in rejoined


def test_collapse_whitespace() -> None:
    text = "Line    with   many    spaces.\n\n\n\nToo   many   newlines.\n  Leading and trailing.  "
    collapsed = collapse_whitespace(text)
    assert "Line with many spaces." in collapsed
    assert "\n\n\n" not in collapsed
    assert "   " not in collapsed


def test_remove_repeated_headers_footers_three_pages() -> None:
    # 3 pages: header and footer appear on all 3 pages (100% >= 50%)
    pages = [
        ExtractedPage(
            number=1,
            text="ACME Corp\nPage 1 body text.\nConfidential",
            headings=["Intro"],
            tables=[],
        ),
        ExtractedPage(
            number=2,
            text="ACME Corp\nPage 2 body text.\nConfidential",
            headings=["Data"],
            tables=[],
        ),
        ExtractedPage(
            number=3,
            text="ACME Corp\nPage 3 body text.\nConfidential",
            headings=["Summary"],
            tables=[],
        ),
    ]
    cleaned = remove_repeated_headers_footers(pages)
    assert len(cleaned) == 3
    for p in cleaned:
        assert "ACME Corp" not in p.text
        assert "Confidential" not in p.text
        assert "body text." in p.text


def test_remove_repeated_headers_footers_50_percent_threshold() -> None:
    # 4 pages: header on pages 1 and 2 (50% >= 50%), footer on only 1 page (25% < 50%)
    pages = [
        ExtractedPage(
            number=1, text="Repeated Header\nPage 1 content.\nFooter 1", headings=[], tables=[]
        ),
        ExtractedPage(
            number=2, text="Repeated Header\nPage 2 content.\nFooter 2", headings=[], tables=[]
        ),
        ExtractedPage(
            number=3, text="Unique Header 3\nPage 3 content.\nFooter 3", headings=[], tables=[]
        ),
        ExtractedPage(
            number=4, text="Unique Header 4\nPage 4 content.\nFooter 4", headings=[], tables=[]
        ),
    ]
    cleaned = remove_repeated_headers_footers(pages)
    assert "Repeated Header" not in cleaned[0].text
    assert "Repeated Header" not in cleaned[1].text
    assert "Unique Header 3" in cleaned[2].text
    assert "Footer 1" in cleaned[0].text


def test_remove_repeated_headers_footers_skipped_for_under_three_pages() -> None:
    # Requirement: only for documents with 3 or more pages
    pages = [
        ExtractedPage(number=1, text="Repeated Header\nPage 1.\nFooter", headings=[], tables=[]),
        ExtractedPage(number=2, text="Repeated Header\nPage 2.\nFooter", headings=[], tables=[]),
    ]
    cleaned = remove_repeated_headers_footers(pages)
    assert "Repeated Header" in cleaned[0].text
    assert "Repeated Header" in cleaned[1].text
    assert "Footer" in cleaned[0].text
    assert "Footer" in cleaned[1].text


def test_clean_document_preserves_page_numbers_and_order() -> None:
    pages = [
        ExtractedPage(number=1, text="Page 1 text", headings=["H1"], tables=[]),
        ExtractedPage(number=2, text="Page 2 text", headings=["H2"], tables=[]),
        ExtractedPage(number=3, text="Page 3 text", headings=["H3"], tables=[]),
    ]
    doc = ExtractedDocument(pages=pages, metadata={"source": "test.pdf"})
    cleaned_doc = clean_document(doc)

    assert len(cleaned_doc.pages) == 3
    assert [p.number for p in cleaned_doc.pages] == [1, 2, 3]
    assert cleaned_doc.metadata == {"source": "test.pdf"}


@pytest.mark.parametrize(
    "sample_input",
    [
        "  Multiple   spaces    between   words  ",
        "Control chars: \x00\x01\x02\x03\x04\x1f End.",
        "Word-\nwrap hyphen-\n  ation here.",
        "Mixed: \u00a0 non-breaking   space and   ligature \ufb01.",
        "Lines\n\n\n\n\nwith blank\n\n\nlines",
        "A regular sentence with state-of-the-art hyphen and no-op words.",
    ],
)
def test_cleaning_properties_and_idempotence(sample_input: str) -> None:
    cleaned = clean_text(sample_input)

    # Property 1: output never contains double spaces
    assert "  " not in cleaned

    # Property 2: output never contains control characters (except \n, \t)
    for ch in cleaned:
        if ch in ("\n", "\t"):
            continue
        category = unicodedata.category(ch)
        assert not category.startswith("C"), f"Found control character {ch!r} in {cleaned!r}"

    # Property 3: idempotence - cleaning twice equals cleaning once
    cleaned_twice = clean_text(cleaned)
    assert cleaned == cleaned_twice


def test_clean_document_idempotence() -> None:
    pages = [
        ExtractedPage(
            number=1,
            text="ACME Corp\nSome multi-\nline docu-\nment with   extra   spaces.\nConfidential",
            headings=["Intro  duction"],
            tables=["Row 1   Col 1"],
        ),
        ExtractedPage(
            number=2,
            text="ACME Corp\nPage two   with   state-of-the-art tech.\nConfidential",
            headings=["Data"],
            tables=[],
        ),
        ExtractedPage(
            number=3,
            text="ACME Corp\nPage three final   summary.\nConfidential",
            headings=["Summary"],
            tables=[],
        ),
    ]
    doc = ExtractedDocument(pages=pages, metadata={"filename": "report.pdf"})
    once = clean_document(doc)
    twice = clean_document(once)

    assert len(once.pages) == len(twice.pages)
    for p1, p2 in zip(once.pages, twice.pages, strict=True):
        assert p1.number == p2.number
        assert p1.text == p2.text
        assert p1.headings == p2.headings
        assert p1.tables == p2.tables
    assert once.metadata == twice.metadata
