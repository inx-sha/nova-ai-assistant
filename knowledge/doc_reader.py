from __future__ import annotations

import fitz  # PyMuPDF
import docx  # python-docx

from core.llm import chat   # <-- add this line

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


# --- add to knowledge/doc_reader.py ---

from core.llm import chat

_VALID_DOC_TYPES = {"resume", "technical_report", "general"}
_CLASSIFY_EXCERPT_CHARS = 800  # ~ first page or so -- doc type is usually obvious early


def classify_doc_type(text: str, source: str = "") -> str:
    """
    Classifies a document as 'resume', 'technical_report', or 'general'
    using the LLM, based on a leading excerpt of its extracted text.
    Defaults to 'general' on any ambiguous/unparseable response --
    never lets a bad classification block ingestion.
    """
    excerpt = text.strip()[:_CLASSIFY_EXCERPT_CHARS]
    if not excerpt:
        return "general"

    prompt = (
        "Classify the following document excerpt as exactly one of these "
        "three words: resume, technical_report, general.\n\n"
        "- resume: a CV/resume listing a person's own skills, education, "
        "work experience, or personal projects as credentials.\n"
        "- technical_report: a document explaining, teaching, or reporting "
        "on a technical subject, project, or system (e.g. a lab report, "
        "tutorial, design doc).\n"
        "- general: anything that doesn't clearly fit either category.\n\n"
        "Respond with ONLY the single classification word, nothing else.\n\n"
        f"--- Document excerpt ---\n{excerpt}"
    )

    try:
        raw = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    except Exception as e:
        print(f"[classify_doc_type] LLM call failed for '{source}': {e} -- defaulting to 'general'")
        return "general"

    cleaned = raw.strip().lower()
    match = next((dt for dt in _VALID_DOC_TYPES if dt in cleaned), "general")

    print(f"[classify_doc_type] source='{source}' raw_response='{raw.strip()}' -> doc_type='{match}'")

    return match