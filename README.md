# NOVA — Offline-First AI Assistant

A local AI assistant that runs on your own machine using a local LLM (Ollama) and a local vector DB (ChromaDB) for retrieval-augmented generation. If it doesn't know something locally, it searches the internet, checks the answer against multiple sources, and saves what it learns so it can answer offline next time. Accessible from a phone on the same network through a simple web UI.

Still in progress. This README covers what's actually built so far.

## Built and working

- Local chat via Ollama (Qwen2.5-3B)
- RAG pipeline: chunk text → embed (nomic-embed-text) → store in ChromaDB → retrieve by similarity
- Router uses the similarity score to decide confidence, not the LLM's own opinion — small models will confidently answer even when retrieval is weak, so the actual number decides, not the model
- Falls back to web search when local knowledge isn't enough. Pulls from multiple sources, cross-checks them, and generates a confidence score based on how much they agree
- SQLite for conversation history, personal facts, known devices, packs, and corrections
- Won't store near-duplicate facts twice (checks similarity before inserting)
- One chunk can belong to multiple categories at once (e.g. something about a medical device's firmware can be tagged health + technology + embedded-systems without storing it twice)
- Tiered storage: `cache` (normal, can be cleaned up later), `pinned` (user said keep this), `pack` (bundled domain knowledge, protected). Pruning function exists and is tested, runs automatically via the scheduler.
- PDF ingestion — upload a PDF, it gets text-extracted, chunked, and stored through the same pipeline as everything else
- Domain knowledge packs: bundles of seed topics (e.g. `embedded_systems`) that get proactively researched via the internet-verification pipeline and stored as `tier="pack"` (protected from pruning), so offline mode has a reliable base of knowledge in a chosen domain instead of only what's been asked about live
- Background scheduler: checks installed packs every 6 hours and re-researches anything older than 7 days, running safely alongside live chat with a shared write lock on the vector store
- Verified corrections: if you tell NOVA it got something wrong, it doesn't just take your word for it — it independently researches the topic and checks whether your correction actually holds up (`verified`, `disputed`, or `unverified`). Only `verified` corrections get fed into future answers.
- Personalized system prompt: confirmed personal preferences and verified corrections get dynamically injected into every conversation, so NOVA adapts its answers to the specific user over time without retraining anything
- Casual conversation handling: greetings and small talk skip the RAG/research pipeline entirely and get a natural, direct reply instead of being treated as knowledge questions
- Simple mobile-friendly web UI, served directly from the FastAPI backend — usable from a phone on the same WiFi network, no app install needed

## Structure

```
nova/
├── app.py # FastAPI: /chat, /knowledge/, /packs/, /corrections/*
├── config.py # settings, overridable via env vars
├── core/
│ ├── llm.py # talks to Ollama
│ ├── memory.py # SQLite: conversations, personal facts, devices, packs, corrections
│ └── router.py # decides how to answer, builds personalized prompt
├── knowledge/
│ ├── store.py # ChromaDB wrapper, thread-safe writes
│ ├── ingest.py # chunking + dedup
│ ├── internet.py # web search, verification, claim-checking
│ ├── pdf_reader.py # PDF text extraction
│ ├── packs.py # domain knowledge pack definitions + installer
│ ├── scheduler.py # background pack-freshness checker
│ └── eval.py # threshold testing
├── ui/
│ └── index.html # mobile-friendly chat interface
└── data/ # db + vector store (gitignored)
```

## Setup

```bash
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama pull nomic-embed-text

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

Open `http://localhost:8000/` for the chat UI, or `http://localhost:8000/docs` for the raw API.

To use from a phone: connect it to the same WiFi as the PC, find the PC's local IP (`ipconfig`), allow port 8000 through the firewall, then visit `http://<pc-ip>:8000/` from the phone's browser.

## Some design notes

- The similarity score decides confidence, not the model. Learned this the hard way after Qwen confidently answered a question with zero local context.
- Internet research never trusts one source. It always pulls a few results and checks if they actually agree before treating anything as verified.
- Corrections work the same way — a user's claim isn't trusted just because they said it. It gets independently researched and compared before it's allowed to influence future answers.
- Only running Chroma with a single worker, and all writes go through a lock, since Chroma's persistent client isn't safe for concurrent access — this matters more now that the background scheduler and live chat can both want to write at the same time.
- The model sometimes blends its own pretrained knowledge into answers even when told to only use retrieved context (tested directly — asked about FreeRTOS queues, got a full code example that wasn't in the stored summary at all). Prompt instructions reduce this but don't fully stop it. Known limitation of small local models, not something fully solved here.
- Personality/tone instructions are followed loosely, not strictly — a 3B model doesn't hold a consistent "voice" as well as something like ChatGPT or Claude does. This is a direct tradeoff of staying fully offline/local instead of using a cloud model — bigger, more consistently personable models need more hardware than what this runs on.

## Not done yet

- RAG thresholds (0.75 / 0.50) are based on a small manual test, not a big benchmark
- Broad/vague questions ("explain X") retrieve weakly since similarity search rewards specific phrasing — known limitation, not yet addressed
- No DOCX/plain-text ingestion yet, only PDF
- No voice, vision, or IoT automation yet

## Bigger picture

This is Phase 1 (basic offline chatbot) plus most of Phase 2 (RAG + internet-verified learning) of a bigger plan that eventually includes voice, automation, and running on dedicated hardware (Jetson Orin Nano + ESP32). Currently using a phone over local WiFi as a stand-in "device" until that hardware is available.