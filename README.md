# NOVA — Offline-First AI Assistant

A local AI assistant that runs on your own machine using a local LLM (Ollama) and a local vector DB (ChromaDB) for retrieval-augmented generation. If it doesn't know something locally, it searches the internet, checks the answer against multiple sources, and saves what it learns so it can answer offline next time.

Still in progress. This README covers what's actually built so far.

## Built and working

- Local chat via Ollama (Qwen2.5-3B)
- RAG pipeline: chunk text → embed (nomic-embed-text) → store in ChromaDB → retrieve by similarity
- Router uses the similarity score to decide confidence, not the LLM's own opinion — small models will confidently answer even when retrieval is weak, so the actual number decides, not the model
- Falls back to web search when local knowledge isn't enough. Pulls from multiple sources, cross-checks them, and generates a confidence score based on how much they agree
- SQLite for conversation history, personal facts, and known devices
- Won't store near-duplicate facts twice (checks similarity before inserting)
- One chunk can belong to multiple categories at once (e.g. something about a medical device's firmware can be tagged health + technology + embedded-systems without storing it twice)
- Tiered storage: `cache` (normal, can be cleaned up later), `pinned` (user said keep this), `pack` (bundled domain knowledge, protected). Pruning function exists and is tested but has to be run manually right now.

## Structure

nova/
├── app.py # FastAPI: /chat, /knowledge/ingest, /knowledge/pin, /knowledge/stats
├── config.py # settings, overridable via env vars
├── core/
│ ├── llm.py # talks to Ollama
│ ├── memory.py # SQLite memory
│ └── router.py # decides how to answer
├── knowledge/
│ ├── store.py # ChromaDB wrapper
│ ├── ingest.py # chunking + dedup
│ ├── internet.py # web search + verification
│ └── eval.py # threshold testing
└── data/ # db + vector store (gitignored)

## Setup

```bash
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama pull nomic-embed-text

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

Then open `http://localhost:8000/docs`.

## Some design notes

- The similarity score decides confidence, not the model. Learned this the hard way after Qwen confidently answered a question with zero local context — turns out asking the model nicely to say "I don't know" isn't reliable enough on its own.
- Internet research never trusts one source. It always pulls a few results and checks if they actually agree before treating anything as verified.
- Only running Chroma with a single worker for now since its persistent client isn't safe for concurrent writes.

## Not done yet

- RAG thresholds (0.75 / 0.50) are based on a small manual test, not a big benchmark
- No scheduler for automatic pruning yet
- No "packs" system yet (pre-loading domain-specific knowledge like embedded systems basics when online, so it's still useful offline)
- No voice, vision, or IoT automation yet

## Bigger picture

This is Phase 1 (basic offline chatbot) plus most of Phase 2 (RAG + internet-verified learning) of a bigger plan that eventually includes voice, automation, and running on dedicated hardware (Jetson Orin Nano + ESP32).