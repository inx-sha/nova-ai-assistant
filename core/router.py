from __future__ import annotations

from dataclasses import dataclass

from core import memory
from core.llm import chat, chat_stream
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
            for c in corrections[:10]
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
    user_message_id: int | None = None
    assistant_message_id: int | None = None


@dataclass
class _PreparedResponse:
    """Everything needed to generate the final answer, computed once and
    shared by both the normal (non-streaming) and streaming code paths --
    avoids duplicating all the retrieval/research/routing logic twice."""
    messages: list[dict]
    mode: str
    sources: list[str]
    filed_elsewhere: bool
    answer_temperature: float | None
    user_message_id: int


def _prepare_response(session_id: str, user_input: str) -> _PreparedResponse:
    ensure_session(session_id)
    session_mode = get_session_mode(session_id)
    user_message_id = memory.add_message(session_id, "user", user_input)
    outcome = None

    if _is_casual(user_input):
        recent = memory.get_recent_messages(session_id, limit=10)
        messages = [{"role": "system", "content": _build_system_prompt()}, *recent,
                    {"role": "user", "content": user_input}]
        return _PreparedResponse(
            messages=messages, mode="casual", sources=[], filed_elsewhere=False,
            answer_temperature=None, user_message_id=user_message_id,
        )

    category_filter = session_mode if session_mode != "general" else None
    hits = knowledge_query(user_input, top_k=RAG_TOP_K, category_filter=category_filter)
    top_similarity = hits[0]["similarity"] if hits else 0.0
    broad_coverage_count = sum(1 for h in hits if h["similarity"] >= RAG_BROAD_COVERAGE_THRESHOLD)

    if top_similarity >= RAG_HIGH_CONFIDENCE:
        mode = "high_confidence"
        relevant_hits = [h for h in hits if h["similarity"] >= RAG_LOW_CONFIDENCE]
        context = _format_context(relevant_hits)
        prompt = (
            f"The context below is verified, stored knowledge -- treat it "
            f"as fact, not as something uncertain. State the answer "
            f"directly. Do not say you lack information if the context "
            f"below answers the question.\n\n"
            f"Context:\n{context}\n\nQuestion: {user_input}"
        )
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
                f"Note: even though you don't normally have real-time or "
                f"up-to-date information, in this case a live web search "
                f"was just performed and verified moments ago -- the "
                f"summary below IS current, real, confirmed information, "
                f"not something you need to be cautious about.\n\n"
                f"Verified summary:\n{outcome.summary}\n\n"
                f"State the answer directly and plainly. Do not say you "
                f"lack information, cannot verify something, or need to "
                f"check elsewhere -- you already have a confirmed answer "
                f"above.\n\nQuestion: {user_input}"
            )

    history_limit = 0 if mode in ("high_confidence", "broad_confidence", "learned_from_internet") else 10
    recent = memory.get_recent_messages(session_id, limit=history_limit) if history_limit else []
    messages = [{"role": "system", "content": _build_system_prompt()}, *recent,
                {"role": "user", "content": user_input}]

    answer_temperature = 0.1 if mode in ("high_confidence", "broad_confidence", "learned_from_internet") else None

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

    filed_elsewhere = mode == "learned_from_internet" and category_filter is not None

    return _PreparedResponse(
        messages=messages, mode=mode, sources=sources, filed_elsewhere=filed_elsewhere,
        answer_temperature=answer_temperature, user_message_id=user_message_id,
    )


def route_query(session_id: str, user_input: str) -> RouteResult:
    prepared = _prepare_response(session_id, user_input)
    answer = chat(prepared.messages, temperature=prepared.answer_temperature)
    assistant_message_id = memory.add_message(session_id, "assistant", answer)

    return RouteResult(
        answer=answer, mode=prepared.mode, sources=prepared.sources,
        filed_elsewhere=prepared.filed_elsewhere,
        user_message_id=prepared.user_message_id,
        assistant_message_id=assistant_message_id,
    )


def route_query_stream(session_id: str, user_input: str):
    """
    Generator version: yields small text chunks as they're generated,
    for progressive display in the UI. The full answer is still saved
    to memory once streaming completes, same as the non-streaming path.
    Yields dicts: {"type": "chunk", "text": ...} while streaming, then
    one final {"type": "done", "mode":..., "sources":..., ...} with
    everything the frontend needs once the answer is complete.
    """
    prepared = _prepare_response(session_id, user_input)
    full_answer = ""

    for chunk in chat_stream(prepared.messages, temperature=prepared.answer_temperature):
        full_answer += chunk
        yield {"type": "chunk", "text": chunk}

    assistant_message_id = memory.add_message(session_id, "assistant", full_answer)

    yield {
        "type": "done",
        "mode": prepared.mode,
        "sources": prepared.sources,
        "filed_elsewhere": prepared.filed_elsewhere,
        "user_message_id": prepared.user_message_id,
        "assistant_message_id": assistant_message_id,
    }


def _format_context(hits: list[dict]) -> str:
    parts = []
    for h in hits:
        src = h["metadata"].get("source", "unknown")
        sim = round(h["similarity"], 3)
        parts.append(f"[source: {src}, similarity: {sim}]\n{h['text']}")
    return "\n\n---\n\n".join(parts)