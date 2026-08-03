from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core import memory
from core.router import route_query
from knowledge.ingest import ingest_text
from knowledge.store import count as knowledge_count
from knowledge.store import set_tier, find_chunks_by_source
import shutil
import tempfile
from fastapi import UploadFile, File, Form
from knowledge.doc_reader import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
from knowledge.packs import install_pack, list_available_packs
from core.memory import get_all_packs
from knowledge.scheduler import start_scheduler, stop_scheduler, refresh_stale_pack_topics
from knowledge.internet import verify_claim
from core.memory import add_correction, get_all_corrections
from core.memory import get_all_sessions, delete_session

app = FastAPI(title="NOVA", version="0.1.0")
app.mount("/static", StaticFiles(directory="ui"), name="static")


@app.get("/")
def serve_ui() -> FileResponse:
    return FileResponse("ui/index.html")

@app.on_event("startup")
def on_startup() -> None:
    memory.init_db()
    start_scheduler()

@app.on_event("shutdown")
def on_shutdown() -> None:
    stop_scheduler()

@app.post("/packs/refresh")
def refresh_packs_endpoint() -> dict:
    return refresh_stale_pack_topics()

class HistoryResponse(BaseModel):
    messages: list[dict]

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    mode: str
    sources: list[str]
    filed_elsewhere: bool = False


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

class InstallPackRequest(BaseModel):
    name: str


class InstallPackResponse(BaseModel):
    pack: str
    topics_researched: int
    topics_total: int
    failed_topics: list[str]

class CorrectionRequest(BaseModel):
    topic: str
    wrong_info: str
    correct_info: str


class CorrectionResponse(BaseModel):
    status: str
    verification_note: str

class SessionInfo(BaseModel):
    session_id: str
    title: str
    last_activity: str


class SessionsListResponse(BaseModel):
    sessions: list[SessionInfo]

class MoveMessageRequest(BaseModel):
    target_session_id: str
    user_message: str
    assistant_message: str


@app.post("/sessions/move_message")
def move_message_endpoint(req: MoveMessageRequest) -> dict:
    from core.memory import ensure_session
    ensure_session(req.target_session_id, mode="general")
    memory.add_message(req.target_session_id, "user", req.user_message)
    memory.add_message(req.target_session_id, "assistant", req.assistant_message)
    return {"moved_to": req.target_session_id}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    result = route_query(req.session_id, req.message)
    return ChatResponse(
        answer=result.answer, mode=result.mode, sources=result.sources,
        filed_elsewhere=result.filed_elsewhere,
    )


@app.post("/knowledge/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest) -> IngestResponse:
    result = ingest_text(
        req.text, req.source, tags=req.tags,
        confidence=req.confidence, categories=req.categories,
    )
    return IngestResponse(**result)

SUPPORTED_DOC_EXTENSIONS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt": extract_text_from_txt,
    ".md": extract_text_from_txt,
}


@app.post("/knowledge/ingest_document", response_model=IngestResponse)
async def ingest_document_endpoint(
    file: UploadFile = File(...),
    tags: str = Form(""),
    confidence: float = Form(1.0),
    categories: str = Form("general"),
) -> IngestResponse:
    filename_lower = file.filename.lower()
    extension = None
    for ext in SUPPORTED_DOC_EXTENSIONS:
        if filename_lower.endswith(ext):
            extension = ext
            break

    if extension is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_DOC_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        extractor = SUPPORTED_DOC_EXTENSIONS[extension]
        text = extractor(tmp_path)
    finally:
        import os
        os.remove(tmp_path)

    if not text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in this file")

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

@app.post("/corrections/add", response_model=CorrectionResponse)
def add_correction_endpoint(req: CorrectionRequest) -> CorrectionResponse:
    status, note = verify_claim(req.topic, req.correct_info)
    add_correction(req.topic, req.wrong_info, req.correct_info, status=status, verification_note=note)
    return CorrectionResponse(status=status, verification_note=note)

@app.post("/packs/install", response_model=InstallPackResponse)
def install_pack_endpoint(req: InstallPackRequest) -> InstallPackResponse:
    try:
        result = install_pack(req.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return InstallPackResponse(**result)

@app.get("/corrections/list")
def list_corrections_endpoint() -> dict:
    return {"corrections": get_all_corrections()}

@app.get("/knowledge/stats")
def knowledge_stats() -> dict:
    return {"total_chunks": knowledge_count()}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/packs/available")
def available_packs() -> dict:
    return {"available_packs": list_available_packs()}


@app.get("/packs/installed")
def installed_packs() -> dict:
    return {"packs": get_all_packs()}

@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history_endpoint(session_id: str) -> HistoryResponse:
    messages = memory.get_recent_messages(session_id, limit=50)
    return HistoryResponse(messages=messages)

@app.get("/sessions", response_model=SessionsListResponse)
def list_sessions_endpoint() -> SessionsListResponse:
    return SessionsListResponse(sessions=get_all_sessions())


@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str) -> dict:
    deleted = delete_session(session_id)
    return {"deleted_messages": deleted}

