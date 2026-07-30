"""
NOVA Phase 1 entrypoint.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core import memory
from core.router import route_query
from knowledge.ingest import ingest_text
from knowledge.store import count as knowledge_count
from knowledge.store import set_tier, find_chunks_by_source
import shutil
import tempfile
from fastapi import UploadFile, File, Form
from knowledge.pdf_reader import extract_text_from_pdf

app = FastAPI(title="NOVA", version="0.1.0")


@app.on_event("startup")
def on_startup() -> None:
    memory.init_db()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    mode: str
    sources: list[str]


class IngestRequest(BaseModel):
    text: str
    source: str
    tags: list[str] = []
    confidence: float = 1.0
    categories: list[str] = ["general"]


class IngestResponse(BaseModel):
    chunks_stored: int
    chunks_skipped_as_duplicate: int

class PinRequest(BaseModel):
    source: str
    tier: str = "pinned"  # or "cache" to un-pin back to default


class PinResponse(BaseModel):
    chunks_updated: int

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    result = route_query(req.session_id, req.message)
    return ChatResponse(answer=result.answer, mode=result.mode, sources=result.sources)


@app.post("/knowledge/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest) -> IngestResponse:
    result = ingest_text(
        req.text, req.source, tags=req.tags,
        confidence=req.confidence, categories=req.categories,
    )
    return IngestResponse(**result)

@app.post("/knowledge/ingest_pdf", response_model=IngestResponse)
async def ingest_pdf_endpoint(
    file: UploadFile = File(...),
    tags: str = Form(""),
    confidence: float = Form(1.0),
    categories: str = Form("general"),
) -> IngestResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text = extract_text_from_pdf(tmp_path)
    finally:
        import os
        os.remove(tmp_path)

    if not text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in this PDF")

    tags_list = [t.strip() for t in tags.split(",") if t.strip()]
    categories_list = [c.strip() for c in categories.split(",") if c.strip()]

    result = ingest_text(
        text, source=file.filename, tags=tags_list,
        confidence=confidence, categories=categories_list,
    )
    return IngestResponse(**result)

@app.post("/knowledge/pin", response_model=PinResponse)
def pin_endpoint(req: PinRequest) -> PinResponse:
    if req.tier not in ("pinned", "cache", "pack"):
        raise HTTPException(status_code=400, detail="tier must be 'pinned', 'cache', or 'pack'")

    chunks = find_chunks_by_source(req.source)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks found for source '{req.source}'")

    updated = 0
    for chunk in chunks:
        if set_tier(chunk["id"], req.tier):
            updated += 1

    return PinResponse(chunks_updated=updated)


@app.get("/knowledge/stats")
def knowledge_stats() -> dict:
    return {"total_chunks": knowledge_count()}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}