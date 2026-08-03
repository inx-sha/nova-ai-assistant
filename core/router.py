from __future__ import annotations

from dataclasses import dataclass

from config import RAG_TOP_K, RAG_HIGH_CONFIDENCE, RAG_LOW_CONFIDENCE
from core import memory
from core.llm import chat
from knowledge.store import query as knowledge_query
from knowledge.internet import research_topic
from knowledge.ingest import ingest_text
from core.memory import get_all_personal_facts, get_verified_corrections, ensure_session, get_session_mode
from config import RAG_TOP_K, RAG_HIGH_CONFIDENCE, RAG_LOW_CONFIDENCE, RAG_BROAD_COVERAGE_THRESHOLD, RAG_BROAD_MIN_CHUNKS

BASE_SYSTEM_PROMPT = (
    "You are NOVA -- a friendly, personal AI assistant, similar in spirit "
    "to how ChatGPT, Claude, or Siri talk to people: warm, natural, and "
    "conversational, not stiff or overly formal. Talk like a knowledgeable "
    "friend, not a corporate support bot. It's fine to have a bit of "
    "personality and warmth in casual conversation.\n\n"
    "For factual/technical questions: answer using ONLY the provided "
    "context when context is given. If the context is marked low-"
    "confidence, say so explicitly before answering. If no context is "
    "provided, say you don't have that information stored locally yet -- "
    "do not guess.\n\n"
    "CRITICAL: Never invent code, syntax, commands, or specific technical "
    "details (function names, APIs, exact values) that are not explicitly "
    "present in the provided context. If the context describes something "
    "only in general terms with no exact code/syntax given, explain the "
    "concept in prose and clearly state that no specific code example was "
    "available in your stored knowledge -- do not fabricate one, even if "
    "it looks plausible. A plausible-looking but invented example is worse "
    "than admitting you don't have one."
)


def _build_system_prompt() -> str:
    parts = [BASE_SYSTEM_PROMPT]

    facts = get_all_personal_facts()
    confirmed_facts = [f for f in facts if f.get("confirmed_by_user")]
    if confirmed_facts:
        facts_text = "\n".join(f"- {f['key']}: {f['value']}" for f in confirmed_facts)
        parts.append(f"\nKnown user preferences/facts:\n{facts_text}")

    corrections = get_verified_corrections()
    if corrections:
        corrections_text = "\n".join(
            f"- On the topic of '{c['topic']}': {c['correct_info']} "
            f"(previously incorrect info to avoid: {c['wrong_info']})"
            for c in corrections[:10]  # cap so the prompt doesn't grow unbounded
        )
        parts.append(f"\nVerified corrections from past conversations -- apply these when relevant:\n{corrections_text}")

    return "\n".join(parts)

CASUAL_PATTERNS = {
    "hi", "hello", "hey", "yo", "sup", "thanks", "thank you", "ok", "okay",
    "cool", "nice", "bye", "goodbye", "good morning", "good night", "test",
    "how are you", "how's it going", "what's up", "whats up", "who are you",
    "what are you", "how are you doing", "what can you do", "help",
}


def _is_casual(text: str) -> bool:
    cleaned = text.strip().lower().rstrip("!?.")
    # normalize punctuation like commas so "hi, how are you" behaves
    # the same as "hi how are you"
    normalized = cleaned.replace(",", " ").replace("-", " ")
    words = normalized.split()

    if cleaned in CASUAL_PATTERNS or normalized in CASUAL_PATTERNS:
        return True

    for prefix in ("hey", "hi", "hello", "ok", "okay", "yo"):
        stripped = cleaned.removeprefix(prefix).strip()
        if stripped in ("nova", ""):
            return True

    casual_words = {"hi", "hello", "hey", "yo", "how", "are", "you", "whats",
                     "what's", "up", "nova", "doing", "there", "sup", "who"}
    if len(words) <= 6 and all(w in casual_words for w in words):
        return True

    return False

@dataclass
class RouteResult:
    answer: str
    mode: str
    sources: list[str]
    filed_elsewhere: bool = False


def route_query(session_id: str, user_input: str) -> RouteResult:
    ensure_session(session_id)
    session_mode = get_session_mode(session_id)
    memory.add_message(session_id, "user", user_input)
    outcome = None

    if _is_casual(user_input):
        recent = memory.get_recent_messages(session_id, limit=10)
        messages = [{"role": "system", "content": _build_system_prompt()}, *recent,
                    {"role": "user", "content": user_input}]
        answer = chat(messages)
        memory.add_message(session_id, "assistant", answer)
        return RouteResult(
        answer=answer, mode=mode, sources=sources,
        filed_elsewhere=(mode == "learned_from_internet" and category_filter is not None),
    )

    category_filter = session_mode if session_mode != "general" else None
    hits = knowledge_query(user_input, top_k=RAG_TOP_K, category_filter=category_filter)
    top_similarity = hits[0]["similarity"] if hits else 0.0
    broad_coverage_count = sum(1 for h in hits if h["similarity"] >= RAG_BROAD_COVERAGE_THRESHOLD)

    if top_similarity >= RAG_HIGH_CONFIDENCE:
        mode = "high_confidence"
        context = _format_context(hits)
        prompt = f"Context:\n{context}\n\nQuestion: {user_input}"
    elif broad_coverage_count >= RAG_BROAD_MIN_CHUNKS:
        mode = "broad_confidence"
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
    messages = [{"role": "system", "content": _build_system_prompt()}, *recent,
                    {"role": "user", "content": user_input}]

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