from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

from core import memory
from core.router import route_query
from knowledge.ingest import ingest_text
from knowledge.store import count as knowledge_count
from knowledge.store import set_tier, find_chunks_by_source
from core.router import route_query_stream
import json as json_module
import shutil
import tempfile
from fastapi import UploadFile, File, Form
from knowledge.doc_reader import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
from knowledge.packs import install_pack, list_available_packs
from core.memory import get_all_packs
from knowledge.scheduler import start_scheduler, stop_scheduler, refresh_stale_pack_topics
from knowledge.internet import verify_claim
from knowledge.backup import create_backup, list_backups
from core.memory import add_correction, get_all_corrections, add_session_file, get_session_files
from core.memory import get_all_sessions, delete_session
from knowledge.packs import list_available_packs
from knowledge.doc_reader import classify_doc_type
from core.memory import set_session_mode as _set_session_mode
from core.memory import (
    set_message_pinned, set_message_starred, get_pinned_messages,
    get_starred_messages_by_mode, edit_message, delete_messages_after,
)

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
    attachment: str | None = None



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
    doc_type: str = "general"


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
    pinned: bool = False
    starred: bool = False
    persona: str | None = None
    persona_subject: str | None = None


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
    persona: str | None = None
    persona_subject: str | None = None

class MessageActionRequest(BaseModel):
    message_id: int


class EditMessageRequest(BaseModel):
    message_id: int
    new_content: str

class RetryRequest(BaseModel):
    session_id: str
    message_id: int
    new_message: str

class ArchiveRequest(BaseModel):
    archived: bool

class PinSessionRequest(BaseModel):
    pinned: bool

class StarSessionRequest(BaseModel):
    starred: bool

class SetModelRequest(BaseModel):
    model: str


@app.post("/sessions/{session_id}/archive")
def archive_session_endpoint(session_id: str, req: ArchiveRequest) -> dict:
    from core.memory import set_session_archived
    set_session_archived(session_id, req.archived)
    return {"session_id": session_id, "archived": req.archived}

@app.post("/sessions/{session_id}/pin")
def pin_session_endpoint(session_id: str, req: PinSessionRequest) -> dict:
    from core.memory import set_session_pinned
    set_session_pinned(session_id, req.pinned)
    return {"session_id": session_id, "pinned": req.pinned}

@app.post("/sessions/{session_id}/star")
def star_session_endpoint(session_id: str, req: StarSessionRequest) -> dict:
    from core.memory import set_session_starred
    set_session_starred(session_id, req.starred)
    return {"session_id": session_id, "starred": req.starred}

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
    result = route_query(req.session_id, req.message, attachment=req.attachment)
    return ChatResponse(
        answer=result.answer, mode=result.mode, sources=result.sources,
        filed_elsewhere=result.filed_elsewhere,
        user_message_id=result.user_message_id,
        assistant_message_id=result.assistant_message_id,
    )

@app.post("/chat/stream")
def chat_stream_endpoint(req: ChatRequest):
    from fastapi.responses import StreamingResponse

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    def event_generator():
        for event in route_query_stream(req.session_id, req.message, attachment=req.attachment):
            yield f"data: {json_module.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/knowledge/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest) -> IngestResponse:
    result = ingest_text(
        req.text, req.source, tags=req.tags,
        confidence=req.confidence, categories=req.categories,
        doc_type=req.doc_type,
    )
    return IngestResponse(**result)

SUPPORTED_DOC_EXTENSIONS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt": extract_text_from_txt,
    ".md": extract_text_from_txt,
}


@app.post("/knowledge/ingest_document", response_model=IngestResponse)
def ingest_document_endpoint(
    file: UploadFile = File(...),
    tags: str = Form(""),
    confidence: float = Form(1.0),
    categories: str = Form("general"),
    doc_type: str | None = Form(None),
    session_id: str = Form(""),
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

    safe_filename = os.path.basename(file.filename)
    saved_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(saved_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    try:
        extractor = SUPPORTED_DOC_EXTENSIONS[extension]
        text = extractor(saved_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to extract text from file: {e}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in this file")

    tags_list = [t.strip() for t in tags.split(",") if t.strip()]
    categories_list = [c.strip() for c in categories.split(",") if c.strip()]

    if doc_type is None:
        doc_type = classify_doc_type(text, source=file.filename)

    result = ingest_text(
        text, source=file.filename, tags=tags_list,
        confidence=confidence, categories=categories_list,
        doc_type=doc_type,
    )

    if session_id.strip():
        from core.memory import ensure_session
        ensure_session(session_id, mode=categories_list[0] if categories_list else "general")
        add_session_file(
            session_id, file.filename, file.filename,
            doc_type=doc_type, categories=", ".join(categories_list)
        )

    return IngestResponse(**result)

@app.get("/sessions/{session_id}/files")
def get_session_files_endpoint(session_id: str) -> dict:
    from knowledge.store import get_all_uploaded_files
    return {
        "files": get_session_files(session_id),
        "session_files": get_session_files(session_id),
        "all_files": get_all_uploaded_files(),
    }

@app.get("/knowledge/files")
def get_knowledge_files_endpoint() -> dict:
    from knowledge.store import get_all_uploaded_files
    return {"files": get_all_uploaded_files()}

@app.get("/files/raw/{filename}")
def get_raw_file_endpoint(filename: str):
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    if os.path.isfile(file_path):
        return FileResponse(file_path, filename=safe_filename, content_disposition_type="inline")

    # Fallback to extracted text from Chroma knowledge base
    from knowledge.store import find_chunks_by_source
    chunks = find_chunks_by_source(filename)
    if chunks:
        full_text = "\n\n".join(c.get("text", "") for c in chunks if c.get("text"))
        if full_text:
            return Response(
                content=full_text,
                media_type="text/plain; charset=utf-8",
                headers={"Content-Disposition": f'inline; filename="{safe_filename}.txt"'},
            )

    raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

@app.get("/files/download/{filename}")
def download_file_endpoint(filename: str):
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    if os.path.isfile(file_path):
        return FileResponse(file_path, filename=safe_filename, content_disposition_type="attachment")

    # Fallback for knowledge chunks
    from knowledge.store import find_chunks_by_source
    chunks = find_chunks_by_source(filename)
    if chunks:
        full_text = "\n\n".join(c.get("text", "") for c in chunks if c.get("text"))
        if full_text:
            return Response(
                content=full_text,
                media_type="text/plain; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{safe_filename}.txt"'},
            )

    raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

@app.get("/files/content/{filename}")
def get_file_content_endpoint(filename: str):
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    has_raw = os.path.isfile(file_path)

    from knowledge.store import find_chunks_by_source
    chunks = find_chunks_by_source(filename)
    content = ""
    if chunks:
        content = "\n\n".join(c.get("text", "") for c in chunks if c.get("text"))
    elif has_raw and (safe_filename.endswith(".txt") or safe_filename.endswith(".md")):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

    return {
        "filename": safe_filename,
        "has_raw_file": has_raw,
        "raw_url": f"/files/raw/{safe_filename}",
        "download_url": f"/files/download/{safe_filename}",
        "chunk_count": len(chunks),
        "content": content,
    }


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

@app.get("/personas")
def get_personas_endpoint() -> dict:
    from core.personas import get_persona_templates
    return {"personas": [p.to_dict() for p in get_persona_templates()]}


@app.post("/sessions/{session_id}/mode")
def set_session_mode_endpoint(session_id: str, req: SetModeRequest) -> dict:
    from core.memory import ensure_session
    ensure_session(session_id, mode=req.mode, persona=req.persona, persona_subject=req.persona_subject)
    _set_session_mode(session_id, req.mode, persona=req.persona, persona_subject=req.persona_subject)
    return {
        "session_id": session_id,
        "mode": req.mode,
        "persona": req.persona,
        "persona_subject": req.persona_subject,
    }

@app.post("/messages/pin")
def pin_message_endpoint(req: MessageActionRequest) -> dict:
    set_message_pinned(req.message_id, True)
    return {"pinned": req.message_id}


@app.post("/messages/unpin")
def unpin_message_endpoint(req: MessageActionRequest) -> dict:
    set_message_pinned(req.message_id, False)
    return {"unpinned": req.message_id}


@app.post("/messages/star")
def star_message_endpoint(req: MessageActionRequest) -> dict:
    set_message_starred(req.message_id, True)
    return {"starred": req.message_id}


@app.post("/messages/unstar")
def unstar_message_endpoint(req: MessageActionRequest) -> dict:
    set_message_starred(req.message_id, False)
    return {"unstarred": req.message_id}

@app.post("/chat/retry")
def retry_endpoint(req: RetryRequest) -> ChatResponse:
    from core.memory import delete_messages_after, edit_message
    edit_message(req.message_id, req.new_message)
    delete_messages_after(req.session_id, req.message_id)
    result = route_query(req.session_id, req.new_message)
    return ChatResponse(
        answer=result.answer, mode=result.mode, sources=result.sources,
        filed_elsewhere=result.filed_elsewhere,
        user_message_id=result.user_message_id,
        assistant_message_id=result.assistant_message_id,
    )

@app.post("/backup/create")
def create_backup_endpoint() -> dict:
    from knowledge.backup import create_backup
    try:
        return create_backup()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup failed: {e}")


@app.get("/backup/list")
def list_backups_endpoint() -> dict:
    from knowledge.backup import list_backups
    return {"backups": list_backups()}

@app.get("/corrections/list")
def list_corrections_endpoint() -> dict:
    return {"corrections": get_all_corrections()}

@app.get("/knowledge/stats")
def knowledge_stats() -> dict:
    return {"total_chunks": knowledge_count()}

@app.get("/knowledge/dashboard")
def knowledge_dashboard_endpoint() -> dict:
    from knowledge.store import get_knowledge_stats
    return get_knowledge_stats()

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
    from core.memory import get_session_history_with_ids
    messages = get_session_history_with_ids(session_id, limit=50)
    return HistoryResponse(messages=messages)

@app.get("/sessions", response_model=SessionsListResponse)
def list_sessions_endpoint(include_archived: bool = False) -> SessionsListResponse:
    return SessionsListResponse(sessions=get_all_sessions(include_archived=include_archived))

@app.get("/modes")
def list_modes_endpoint() -> dict:
    from core.memory import get_all_distinct_modes
    pack_modes = list_available_packs()
    custom_modes = set(get_all_distinct_modes())
    all_modes = sorted(custom_modes | set(pack_modes) | {"general"})
    all_modes.remove("general")
    return {"modes": ["general"] + all_modes}
@app.get("/sessions/{session_id}/pinned")
def get_pinned_endpoint(session_id: str) -> dict:
    return {"pinned": get_pinned_messages(session_id)}


@app.get("/starred")
def get_starred_endpoint() -> dict:
    return {"starred_by_mode": get_starred_messages_by_mode()}


@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str) -> dict:
    deleted = delete_session(session_id)
    return {"deleted_messages": deleted}


def detect_gpu_vram_mb() -> int | None:
    """Detects total GPU VRAM in MB via nvidia-smi if available, returns None on failure."""
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0:
            lines = res.stdout.strip().splitlines()
            if lines:
                return int(lines[0].strip())
    except Exception:
        pass
    return None


@app.get("/models")
def list_models_endpoint() -> dict:
    import httpx
    from config import OLLAMA_HOST, LLM_MODEL
    from core.memory import get_current_model, set_current_model, get_setting

    try:
        resp = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to Ollama ({e}). Ensure Ollama is running.",
        )

    raw_models = data.get("models", [])
    filtered_models = []
    for m in raw_models:
        name = m.get("name") or m.get("model", "")
        if not name:
            continue
        capabilities = m.get("capabilities", [])
        family = m.get("details", {}).get("family", "")
        # Filter out embedding-only models
        if capabilities == ["embedding"] or "nomic-embed" in name.lower() or family == "nomic-bert":
            continue

        raw_size = m.get("size", 0)
        size_gb = round(raw_size / (1024 ** 3), 1)
        param_size = m.get("details", {}).get("parameter_size", "")
        quant = m.get("details", {}).get("quantization_level", "")

        filtered_models.append({
            "name": name,
            "size": raw_size,
            "size_gb": size_gb,
            "parameter_size": param_size,
            "quantization": quant,
            "family": family,
        })

    vram_mb = detect_gpu_vram_mb()
    persisted_model = get_setting("llm_model")

    if not persisted_model and filtered_models:
        available_names = [m["name"] for m in filtered_models]
        if vram_mb and vram_mb <= 4500:
            # Pick a model that fits comfortably in ~4GB VRAM
            small_model = next(
                (m["name"] for m in filtered_models if m["size_gb"] <= 4.0 or "4b" in m["name"].lower() or "3b" in m["name"].lower()),
                None
            )
            chosen_model = small_model or (LLM_MODEL if LLM_MODEL in available_names else filtered_models[0]["name"])
        else:
            chosen_model = LLM_MODEL if LLM_MODEL in available_names else filtered_models[0]["name"]

        set_current_model(chosen_model)
        current_model = chosen_model
    else:
        current_model = get_current_model()

    return {
        "models": filtered_models,
        "current_model": current_model,
        "vram_mb": vram_mb,
    }


@app.post("/models")
@app.post("/models/set")
def set_model_endpoint(req: SetModelRequest) -> dict:
    from core.memory import set_current_model
    model_name = req.model.strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="model name cannot be empty")
    set_current_model(model_name)
    return {"status": "success", "model": model_name}