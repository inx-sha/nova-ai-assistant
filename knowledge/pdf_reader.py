from __future__ import annotations

import fitz  # PyMuPDF


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