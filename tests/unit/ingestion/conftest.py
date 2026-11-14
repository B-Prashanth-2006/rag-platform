"""Fixture generators for ingestion tests.

All fixtures are generated programmatically in code — no binary files are checked in.
"""

from __future__ import annotations

import io

import pymupdf as fitz
from docx import Document as DocxDocument
from PIL import Image, ImageDraw


def generate_multipage_pdf_with_headers_footers() -> bytes:
    """Create a 3-page PDF with repeated header/footer lines and body text with headings."""
    doc = fitz.open()

    for page_num in range(1, 4):
        page = doc.new_page(width=595, height=842)  # A4 size

        # Repeated header at top
        page.insert_text(fitz.Point(50, 40), "ACME Corporation Annual Report", fontsize=10)

        # Heading (larger font size)
        page.insert_text(fitz.Point(50, 100), f"Section {page_num}: Overview", fontsize=18)

        # Body text (regular font size)
        body_text = (
            f"This is the detailed body text for page {page_num}. "
            "It contains operational data and financial summary metrics for the fiscal year."
        )
        page.insert_text(fitz.Point(50, 140), body_text, fontsize=11)

        # Repeated footer at bottom
        page.insert_text(fitz.Point(50, 800), "Confidential and Proprietary", fontsize=9)

    pdf_bytes: bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def generate_pdf_with_table() -> bytes:
    """Create a PDF containing a structured table."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    page.insert_text(fitz.Point(50, 50), "Quarterly Financial Results", fontsize=16)

    # Draw table borders so PyMuPDF find_tables recognizes it
    # Outer rectangle
    page.draw_rect(fitz.Rect(50, 80, 450, 180), color=(0, 0, 0), width=1)
    # Horizontal lines
    page.draw_line(fitz.Point(50, 110), fitz.Point(450, 110), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(50, 145), fitz.Point(450, 145), color=(0, 0, 0), width=1)
    # Vertical line separating 2 columns
    page.draw_line(fitz.Point(250, 80), fitz.Point(250, 180), color=(0, 0, 0), width=1)

    # Insert cell texts
    page.insert_text(fitz.Point(60, 100), "Quarter", fontsize=11)
    page.insert_text(fitz.Point(260, 100), "Revenue", fontsize=11)
    page.insert_text(fitz.Point(60, 130), "Q1 2026", fontsize=11)
    page.insert_text(fitz.Point(260, 130), "$1.2M", fontsize=11)
    page.insert_text(fitz.Point(60, 165), "Q2 2026", fontsize=11)
    page.insert_text(fitz.Point(260, 165), "$1.5M", fontsize=11)

    pdf_bytes: bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def generate_scanned_image_pdf() -> bytes:
    """Create an image-only PDF with rasterized text (no text layer) to trigger OCR."""
    img = Image.new("RGB", (600, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 80), "SCANNED INVOICE 99482", fill="black")

    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    doc = fitz.open()
    page = doc.new_page(width=600, height=200)
    page.insert_image(page.rect, stream=img_bytes)

    pdf_bytes: bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def generate_docx_with_headings_and_table() -> bytes:
    """Create a DOCX file containing headings, paragraphs, and a table."""
    doc = DocxDocument()
    doc.add_heading("Docx Main Document Title", level=1)
    doc.add_paragraph("This is an introductory paragraph describing the document.")
    doc.add_heading("Subsection 1.1", level=2)
    doc.add_paragraph("Additional paragraph text under section 1.1.")

    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Col A"
    table.cell(0, 1).text = "Col B"
    table.cell(1, 0).text = "Val 1"
    table.cell(1, 1).text = "Val 2"

    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


def generate_html_with_scripts_and_nav() -> bytes:
    """Create an HTML document containing script, style, nav, headings, and a table."""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Sample Document</title>
    <script>alert('malicious or extraneous script');</script>
    <style>body { background-color: #fff; font-family: sans-serif; }</style>
</head>
<body>
    <nav>
        <a href="#home">Home</a>
        <a href="#about">About</a>
    </nav>
    <h1>Page Main Heading</h1>
    <p>Introductory paragraph in HTML.</p>
    <h2>Secondary Heading</h2>
    <table>
        <thead>
            <tr><th>Item</th><th>Price</th></tr>
        </thead>
        <tbody>
            <tr><td>Widget</td><td>$10</td></tr>
            <tr><td>Gadget</td><td>$20</td></tr>
        </tbody>
    </table>
    <p>Closing paragraph.</p>
</body>
</html>
"""
    return html_content.encode("utf-8")


def generate_markdown() -> bytes:
    """Create a Markdown document with headings, paragraphs, and tables."""
    md_content = """# Markdown Project Report

This is a paragraph introducing the report.

## Methodology

We evaluated several approaches.

| Model | Accuracy |
| --- | --- |
| Model A | 92% |
| Model B | 95% |

### Discussion

Further improvements are expected.
"""
    return md_content.encode("utf-8")


def generate_txt_odd_encoding() -> bytes:
    """Create a text file encoded with ISO-8859-1 (Latin-1) containing accented characters."""
    text = "Rapport Financier: café, naïve, résumé et façade de l'entreprise.\n"
    return text.encode("latin-1")


def generate_empty_file() -> bytes:
    """Return empty bytes."""
    return b""


def generate_corrupt_pdf() -> bytes:
    """Return corrupted PDF bytes that cannot be opened as a valid PDF."""
    return b"%PDF-1.7 corrupt invalid header\x00\xffrandom garbage bytes without xref"
