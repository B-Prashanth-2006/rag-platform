"""Tests verifying that fixture generators work and produce non-empty byte streams."""

from __future__ import annotations

from .conftest import (
    generate_corrupt_pdf,
    generate_docx_with_headings_and_table,
    generate_empty_file,
    generate_html_with_scripts_and_nav,
    generate_markdown,
    generate_multipage_pdf_with_headers_footers,
    generate_pdf_with_table,
    generate_scanned_image_pdf,
    generate_txt_odd_encoding,
)


def test_fixture_generators() -> None:
    pdf_multi = generate_multipage_pdf_with_headers_footers()
    assert pdf_multi.startswith(b"%PDF")

    pdf_table = generate_pdf_with_table()
    assert pdf_table.startswith(b"%PDF")

    pdf_scanned = generate_scanned_image_pdf()
    assert pdf_scanned.startswith(b"%PDF")

    docx_data = generate_docx_with_headings_and_table()
    assert len(docx_data) > 0
    assert docx_data.startswith(b"PK")  # zip archive format for docx

    html_data = generate_html_with_scripts_and_nav()
    assert b"Page Main Heading" in html_data

    md_data = generate_markdown()
    assert b"# Markdown Project Report" in md_data

    txt_data = generate_txt_odd_encoding()
    assert len(txt_data) > 0

    empty_data = generate_empty_file()
    assert len(empty_data) == 0

    corrupt_data = generate_corrupt_pdf()
    assert corrupt_data.startswith(b"%PDF")
