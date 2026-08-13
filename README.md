# NOVA — Offline-First AI Assistant

A local AI assistant that runs on your own machine using a local LLM (Ollama) and a local vector DB (ChromaDB) for retrieval-augmented generation. If it doesn't know something locally, it searches the internet, checks the answer against multiple sources, and saves what it learns so it can answer offline next time. Accessible from a phone on the same network through a simple web UI.

Still in progress. This README covers what's actually built so far.

## Built and working

- Local chat via Ollama (Qwen3.5)
- RAG pipeline: chunk text → embed (nomic-embed-text) → store in ChromaDB → retrieve by similarity
- Router uses the similarity score to decide confidence, not the LLM's own opinion — small models will confidently answer even when retrieval is weak, so the actual number decides, not the model
- Falls back to web search when local knowledge isn't enough. Pulls from multiple sources, cross-checks them, and generates a confidence score based on how much they agree
- SQLite for conversation history, personal facts, known devices, packs, and corrections
- Won't store near-duplicate facts twice (checks similarity before inserting)
- One chunk can belong to multiple categories at once (e.g. something about a medical device's firmware can be tagged health + technology + embedded-systems without storing it twice)
- Tiered storage: `cache` (normal, can be cleaned up later), `pinned` (user said keep this), `pack` (bundled domain knowledge, protected). Pruning function exists and is tested, runs automatically via the scheduler.
- Document ingestion — upload a PDF, DOCX, or plain text/markdown file through `/knowledge/ingest_document`; it gets text-extracted (tables included for DOCX), chunked, and stored through the same pipeline as everything else
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
│ ├── doc_reader.py # PDF text extraction
│ ├── packs.py # domain knowledge pack definitions + installer
│ ├── scheduler.py # background pack-freshness checker
│ └── eval.py # threshold testing
├── ui/
│ └── index.html # mobile-friendly chat interface
└── data/ # db + vector store (gitignored)
```

## Setup

```bash
ollama pull qwen3.5:latest
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
- The model doesn't reliably follow strict instructions, confirmed across several
  separate tests, not just a one-off:
  - Asked about FreeRTOS queues/semaphores — got a fabricated code example
    not present in the stored context, at one point using invented
    (non-existent) syntax, at another point using real FreeRTOS API calls
    pulled from pretrained knowledge rather than stored data
  - Asked "what is the capital of Mongolia" with nothing stored locally —
    answered from pretrained knowledge instead of admitting no local
    answer, despite an explicit instruction not to
  - Asked to explain the SAP-1 CPU with real project-report context
    provided — answered with a generic, wrong, 32-bit/register-based CPU
    explanation from pretrained knowledge, effectively ignoring the actual
    SAP-1-specific context it was given
  - A stored "prefers concise answers" preference is visible in the
    prompt but not consistently followed — verbose answers still happen

  Multiple rounds of prompt tightening reduced but didn't eliminate this.
  Concluded this is a genuine ceiling of a 3B-parameter model's
  instruction-following ability, not a prompt-wording problem — a larger
  model would likely handle this better, but that needs more VRAM than
  this hardware has. Documented rather than chased further.
  - **Deep dive: time-sensitive/current-events questions specifically.** A
  more rigorous investigation (not just one failed test) into why NOVA
  couldn't answer "what is the release date of Spider-Man: Brand New Day"
  even with a correct, verified answer already stored locally:
  1. Found and fixed a real bug first: a confidence-parsing regex only
     matched the literal string "CONFIDENCE:", but the LLM sometimes
     produces variants ("CONFIANCE", "CONFINENCE") — when unmatched, the
     raw unparsed response (including garbled label text) got stored as
     the "clean" summary, corrupting several stored knowledge chunks.
     Fixed with a more forgiving regex; corrupted chunks manually removed.
  2. Found and fixed a second real bug: `high_confidence` mode included
     all top-5 retrieved chunks in the prompt regardless of relevance,
     so a single correct match got diluted by 4 unrelated low-similarity
     chunks (weather forecasts, an unrelated celebrity, etc). Fixed by
     filtering to only chunks above the low-confidence threshold.
  3. Found and fixed a third real bug: conversation history was included
     in every prompt, meaning an earlier failed/wrong answer in the same
     session got fed back as context on the next turn, reinforcing the
     same mistake. Fixed by excluding history for high-confidence answer
     paths, where the current context is already self-contained.
  4. After all three fixes, the failure persisted. Tested temperature
     0.1 (reduced sampling randomness) — no improvement. Tested
     increasingly explicit, repeated prompt instructions explicitly
     telling the model the provided information was current and verified
     — no improvement.
  5. Swapped the entire model (Qwen2.5-3B → Phi-3.5-mini) as a final
     test, keeping everything else identical. Result: consistent,
     repeatable failure across 5/5 trials on both models -- Phi-3.5 even
     cited its own training cutoff ("as of early 2023") and fabricated an
     unrelated real movie's release date, rather than using the provided
     current information.

  Conclusion: this is not a prompt-engineering gap. It's evidence that
  small instruction-tuned models have a strongly reinforced training
  behavior around deferring to their own knowledge cutoff for
  date/current-events questions, which in-context correction can't
  reliably override at this model scale — consistent across two
  independent model families. The three real bugs found along the way
  (confidence parsing, irrelevant context dilution, history poisoning)
  were fixed and are genuine improvements; the residual failure on this
  specific question type is a documented, tested, and accepted
  limitation, not an unexamined one.
- Personality/tone instructions are followed loosely for the same reason above.
  This is a direct tradeoff of staying fully offline/local on modest hardware
  instead of using a cloud model.

## Not done yet

- RAG thresholds (0.75 / 0.50) are based on a small manual test, not a big benchmark
- Broad/vague questions ("explain X") retrieve weakly since similarity search rewards specific phrasing — known limitation, not yet addressed
- No voice, vision, or IoT automation yet

## Bigger picture

This is Phase 1 (basic offline chatbot) plus most of Phase 2 (RAG + internet-verified learning) of a bigger plan that eventually includes voice, automation, and running on dedicated hardware (Jetson Orin Nano + ESP32). Currently using a phone over local WiFi as a stand-in "device" until that hardware is available.