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
from knowledge.packs import list_available_packs
from core.memory import set_session_mode as _set_session_mode

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
    user_message_id: int | None = None
    assistant_message_id: int | None = None

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
    mode: str = "general"
    archived: bool = False


class SessionsListResponse(BaseModel):
    sessions: list[SessionInfo]

class MoveMessageRequest(BaseModel):
    target_session_id: str
    user_message: str
    assistant_message: str
    source_user_message_id: int | None = None
    source_assistant_message_id: int | None = None

class SetModeRequest(BaseModel):
    mode: str

@app.post("/sessions/move_message")
def move_message_endpoint(req: MoveMessageRequest) -> dict:
    from core.memory import ensure_session, delete_messages_by_ids, set_session_mode
    ensure_session(req.target_session_id, mode="general")
    set_session_mode(req.target_session_id, "general")
    memory.add_message(req.target_session_id, "user", req.user_message)
    memory.add_message(req.target_session_id, "assistant", req.assistant_message)

    ids_to_remove = [
        i for i in (req.source_user_message_id, req.source_assistant_message_id)
        if i is not None
    ]
    delete_messages_by_ids(ids_to_remove)

    return {"moved_to": req.target_session_id, "removed_from_source": len(ids_to_remove)}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    result = route_query(req.session_id, req.message)
    return ChatResponse(
        answer=result.answer, mode=result.mode, sources=result.sources,
        filed_elsewhere=result.filed_elsewhere,
        user_message_id=result.user_message_id,
        assistant_message_id=result.assistant_message_id,
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

@app.post("/sessions/{session_id}/mode")
def set_session_mode_endpoint(session_id: str, req: SetModeRequest) -> dict:
    from core.memory import ensure_session
    ensure_session(session_id)
    _set_session_mode(session_id, req.mode)
    return {"session_id": session_id, "mode": req.mode}

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

@app.get("/modes")
def list_modes_endpoint() -> dict:
    from core.memory import get_all_distinct_modes
    pack_modes = list_available_packs()
    custom_modes = set(get_all_distinct_modes())
    all_modes = sorted(custom_modes | set(pack_modes) | {"general"})
    all_modes.remove("general")
    return {"modes": ["general"] + all_modes}

@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str) -> dict:
    deleted = delete_session(session_id)
    return {"deleted_messages": deleted}

