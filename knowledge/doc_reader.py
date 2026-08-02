from __future__ import annotations

import fitz  # PyMuPDF
import docx  # python-docx


def extract_text_from_pdf(file_path: str) -> str:
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


def extract_text_from_docx(file_path: str) -> str:

    document = docx.Document(file_path)
    parts = []

    for para in document.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    return "\n\n".join(parts)


def extract_text_from_txt(file_path: str) -> str:
    """Plain text / markdown files -- just read them directly."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()