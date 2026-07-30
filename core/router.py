"""
Router: decides how to answer a user query.

Key design decision: the SIMILARITY SCORE decides confidence, not the
LLM's own judgment. Small models tend to answer confidently even from
weak retrieval, so we measure trust with embeddings math and only ask
the model to generate text over what's already been judged reliable.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import RAG_TOP_K, RAG_HIGH_CONFIDENCE, RAG_LOW_CONFIDENCE
from core import memory
from core.llm import chat
from knowledge.store import query as knowledge_query
from knowledge.internet import research_topic
from knowledge.ingest import ingest_text

SYSTEM_PROMPT = (
    "You are NOVA, a private, offline-first AI assistant. Answer using "
    "ONLY the provided context when context is given. If the context is "
    "marked low-confidence, say so explicitly before answering. If no "
    "context is provided, say you don't have that information stored "
    "locally yet -- do not guess."
)


@dataclass
class RouteResult:
    answer: str
    mode: str  # "high_confidence" | "low_confidence" | "no_local_answer" | "learned_from_internet"
    sources: list[str]


def route_query(session_id: str, user_input: str) -> RouteResult:
    memory.add_message(session_id, "user", user_input)
    outcome = None

    hits = knowledge_query(user_input, top_k=RAG_TOP_K)
    top_similarity = hits[0]["similarity"] if hits else 0.0

    if top_similarity >= RAG_HIGH_CONFIDENCE:
        mode = "high_confidence"
        context = _format_context(hits)
        prompt = f"Context:\n{context}\n\nQuestion: {user_input}"
    elif top_similarity >= RAG_LOW_CONFIDENCE:
        mode = "low_confidence"
        context = _format_context(hits)
        prompt = (
            f"[LOW CONFIDENCE CONTEXT -- flag uncertainty to the user]\n"
            f"Context:\n{context}\n\nQuestion: {user_input}"
        )
    else:
        outcome = research_topic(user_input)

        if outcome is None:
            # No internet, or search/fetch genuinely found nothing usable.
            mode = "no_local_answer"
            prompt = (
                "No relevant local context was found for this question, "
                "and internet research did not turn up anything usable. "
                "You MUST NOT answer using your own general knowledge. "
                "Respond with exactly: \"I don't have information about "
                "this stored locally, and couldn't find anything reliable "
                "online right now.\"\n\n"
                f"Question: {user_input}"
            )
        else:
            # Store what was just learned so it's available offline next time.
            ingest_text(
                outcome.summary,
                source=f"web_research:{outcome.sources[0]}",
                tags=["web-research"],
                confidence=outcome.confidence,
                categories=["general"],
            )
            mode = "learned_from_internet"
            prompt = (
                f"You just researched this online. Here is a verified "
                f"summary (confidence: {outcome.confidence}):\n\n"
                f"{outcome.summary}\n\n"
                f"Answer the user's question using this summary. If "
                f"confidence is below 0.6, mention you're not fully "
                f"certain.\n\nQuestion: {user_input}"
            )

    recent = memory.get_recent_messages(session_id, limit=10)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *recent,
                {"role": "user", "content": prompt}]

    answer = chat(messages)
    memory.add_message(session_id, "assistant", answer)

    if mode == "learned_from_internet":
        sources = outcome.sources
    elif mode == "no_local_answer":
        sources = []
    else:
        sources = sorted({
            h["metadata"].get("source", "unknown")
            for h in hits
            if h["similarity"] >= RAG_LOW_CONFIDENCE
        })
    return RouteResult(answer=answer, mode=mode, sources=sources)


def _format_context(hits: list[dict]) -> str:
    parts = []
    for h in hits:
        src = h["metadata"].get("source", "unknown")
        sim = round(h["similarity"], 3)
        parts.append(f"[source: {src}, similarity: {sim}]\n{h['text']}")
    return "\n\n---\n\n".join(parts)