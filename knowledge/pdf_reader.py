"""
PDF text extraction. Kept separate from ingest.py so the chunking logic
stays format-agnostic -- this file's only job is "PDF in, plain text out."
"""
from __future__ import annotations

import fitz  # PyMuPDF


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts all readable text from a PDF, page by page, joined with
    double newlines so ingest.py's paragraph-based chunker treats each
    page boundary as a natural break point.
    """
    doc = fitz.open(file_path)
    pages = []
    try:
        for page in doc:
            text = page.get_text().strip()
            if text:
                pages.append(text)
    finally:
        doc.close()

    return "\n\n".join(pages)