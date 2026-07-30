import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- LLM ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("NOVA_LLM_MODEL", "qwen2.5:3b-instruct-q4_K_M")
LLM_CONTEXT_WINDOW = int(os.getenv("NOVA_LLM_CTX", "4096"))
LLM_TEMPERATURE = float(os.getenv("NOVA_LLM_TEMP", "0.3"))

# --- Embeddings ---
EMBED_MODEL = os.getenv("NOVA_EMBED_MODEL", "nomic-embed-text")

# --- Storage ---
SQLITE_PATH = os.getenv("NOVA_SQLITE_PATH", str(DATA_DIR / "nova.db"))
CHROMA_PATH = os.getenv("NOVA_CHROMA_PATH", str(DATA_DIR / "chroma"))
CHROMA_COLLECTION = os.getenv("NOVA_CHROMA_COLLECTION", "knowledge")

# --- Retrieval thresholds ---
RAG_TOP_K = int(os.getenv("NOVA_RAG_TOP_K", "5"))
RAG_HIGH_CONFIDENCE = float(os.getenv("NOVA_RAG_HIGH_CONF", "0.75"))
RAG_LOW_CONFIDENCE = float(os.getenv("NOVA_RAG_LOW_CONF", "0.50"))

# --- Chunking ---
CHUNK_SIZE_TOKENS = int(os.getenv("NOVA_CHUNK_SIZE", "512"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("NOVA_CHUNK_OVERLAP", "50"))