from core.router import _prepare_response
from core.memory import get_session_mode, ensure_session
from knowledge.store import find_chunks_by_source

session_id = "debug-test-general"
ensure_session(session_id)
print("Session mode:", get_session_mode(session_id))

# What category did the actual uploaded chunks get tagged with?
import chromadb
from knowledge.store import get_collection
all_chunks = get_collection().get()
for meta in all_chunks["metadatas"]:
    if "SAP1" in meta.get("source", ""):
        print("Chunk category:", meta.get("categories"))
        break

prepared = _prepare_response(session_id, "explain sap-1")
print("\nMode:", prepared.mode)
print("Sources:", prepared.sources)