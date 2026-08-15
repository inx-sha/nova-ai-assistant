from __future__ import annotations

import math
from dataclasses import dataclass

from core import memory
from core.llm import chat, chat_stream, embed
from core.personas import get_persona
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


def _build_system_prompt(session_id: str | None = None) -> str:
    parts = [BASE_SYSTEM_PROMPT]

    if session_id:
        persona_id, persona_subject = memory.get_session_persona(session_id)
        template = get_persona(persona_id)
        if template:
            subject_clause = f" in {persona_subject}" if (persona_subject and persona_subject.strip()) else ""
            persona_text = template.system_prompt.replace("{subject_clause}", subject_clause)
            parts.append(f"\nPersona Directives ({template.name}):\n{persona_text}")

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

RESUME_PATTERNS = {
    "resume", "cv", "my experience", "my education", "my skills",
    "my background", "my work experience", "my projects", "my qualifications",
    "selected projects", "about me", "my profile",
}

RESUME_ANCHOR_PHRASES = [
    "questions about my work experience, education, skills, resume, CV",
    "what is my educational background and university degrees",
    "tell me about my professional job experience and career history",
    "list my technical skills, qualifications, and personal profile",
    "what projects have I worked on in my past jobs or portfolio",
]
RESUME_SIMILARITY_THRESHOLD = 0.60

_cached_resume_anchor_embeddings: list[list[float]] | None = None


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def _get_resume_anchor_embeddings() -> list[list[float]]:
    global _cached_resume_anchor_embeddings
    if _cached_resume_anchor_embeddings is None:
        _cached_resume_anchor_embeddings = [embed(phrase) for phrase in RESUME_ANCHOR_PHRASES]
    return _cached_resume_anchor_embeddings


def _is_resume_query(text: str) -> bool:
    cleaned = text.strip().lower()
    # 1. Fast-path short-circuit keyword matching
    if any(p in cleaned for p in RESUME_PATTERNS):
        return True

    # 2. Embedding cosine similarity check against anchor phrases
    try:
        query_vec = embed(cleaned)
        anchor_vecs = _get_resume_anchor_embeddings()
        max_sim = max((_cosine_similarity(query_vec, av) for av in anchor_vecs), default=0.0)
        return max_sim >= RESUME_SIMILARITY_THRESHOLD
    except Exception:
        return False


def _find_target_document(session_id: str, user_input: str, attachment: str | None = None) -> tuple[str | None, list[dict]]:
    """
    Determines if the query targets a specific document (e.g., attached file,
    uploaded session file, or CV/resume reference). If found, returns (source_name, chunks).
    """
    from knowledge.store import find_chunks_by_source, get_all_uploaded_files
    cleaned = user_input.strip().lower()

    # 1. Direct attachment parameter in current message
    if attachment:
        chunks = find_chunks_by_source(attachment)
        if chunks:
            return attachment, chunks
        for f in get_all_uploaded_files():
            fn = f.get("filename", "")
            if fn.lower() == attachment.lower() or attachment.lower() in fn.lower():
                c = find_chunks_by_source(fn)
                if c:
                    return fn, c

    # 2. Uploaded files in the current session
    session_files = memory.get_session_files(session_id)
    if session_files:
        is_doc_reference = any(k in cleaned for k in (
            "this cv", "my cv", "the cv", "this resume", "my resume", "the resume",
            "this file", "this document", "the document", "in the pdf", "the pdf",
            "uploaded file", "this paper", "the report", "above cv", "attached file",
            "this attachment", "project section", "experience section", "skills section",
            "education section", "summarize this", "explain this"
        ))

        # Check for specific filename match
        for sf in session_files:
            fn = sf.get("filename", "")
            fn_base = fn.rsplit(".", 1)[0].lower()
            if fn_base and fn_base in cleaned:
                chunks = find_chunks_by_source(fn)
                if chunks:
                    return fn, chunks

        if is_doc_reference or len(session_files) == 1:
            target_fn = session_files[-1].get("filename", "")
            if target_fn:
                chunks = find_chunks_by_source(target_fn)
                if chunks:
                    return target_fn, chunks

    # 3. Global CV/Resume query or file reference across knowledge base
    if _is_resume_query(cleaned) or any(k in cleaned for k in ("in the cv", "in this cv", "from the cv", "above cv", "in my cv", "of this cv", "in my resume", "the cv", "this cv", "my cv")):
        all_files = get_all_uploaded_files()
        resume_files = [
            f for f in all_files
            if f.get("doc_type") == "resume" or "cv" in f.get("filename", "").lower() or "resume" in f.get("filename", "").lower() or "inshaf" in f.get("filename", "").lower()
        ]
        if resume_files:
            target_fn = resume_files[0].get("filename", "")
            chunks = find_chunks_by_source(target_fn)
            if chunks:
                return target_fn, chunks

    return None, []


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
    static_answer: str | None = None


def _prepare_response(session_id: str, user_input: str, attachment: str | None = None) -> _PreparedResponse:
    ensure_session(session_id)
    session_mode = get_session_mode(session_id)
    user_message_id = memory.add_message(session_id, "user", user_input, attachment=attachment)
    outcome = None

    # Check if session has a persona that requires a subject, but none is set yet
    persona_id, persona_subject = memory.get_session_persona(session_id)
    persona_tmpl = get_persona(persona_id)
    if persona_tmpl and persona_tmpl.requires_subject and not (persona_subject and persona_subject.strip()):
        prompt_q = persona_tmpl.subject_prompt or "Please specify which subject or topic you'd like to focus on."
        return _PreparedResponse(
            messages=[],
            mode="needs_subject",
            sources=[],
            filed_elsewhere=False,
            answer_temperature=None,
            user_message_id=user_message_id,
            static_answer=prompt_q,
        )

    if _is_casual(user_input):
        recent = memory.get_recent_messages(session_id, limit=5)
        messages = [
            {"role": "system", "content": _build_system_prompt(session_id)},
            *recent,
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": "<think>\n</think>"}
        ]
        return _PreparedResponse(
            messages=messages, mode="casual", sources=[], filed_elsewhere=False,
            answer_temperature=None, user_message_id=user_message_id,
        )

    # Check for document-specific targeting (attached file, session file, or CV/resume query)
    target_source, doc_chunks = _find_target_document(session_id, user_input, attachment=attachment)
    if target_source and doc_chunks:
        mode = "high_confidence"
        sources = [target_source]
        full_doc_text = "\n\n".join(c["text"] for c in doc_chunks)
        prompt = (
            f"The context below is the exact, verified document content from '{target_source}'. "
            f"Treat this document content as definitive fact.\n"
            f"When asked about sections, projects, experiences, skills, education, or details, "
            f"carefully review the entire document text and provide a complete, clear, comprehensive answer "
            f"listing ALL relevant items from the document without omitting any entries or inventing information.\n\n"
            f"Document Content ({target_source}):\n{full_doc_text}\n\n"
            f"Question: {user_input}"
        )
        # Limit recent context to last 2 relevant turns to avoid token bloat/slow prefill
        recent = memory.get_recent_messages(session_id, limit=2)
        clean_recent = [r for r in recent if r.get("content", "").strip() and r["content"] != user_input]
        messages = [{"role": "system", "content": _build_system_prompt(session_id)}, *clean_recent,
                    {"role": "user", "content": prompt}]
        return _PreparedResponse(
            messages=messages, mode=mode, sources=sources, filed_elsewhere=False,
            answer_temperature=0.1, user_message_id=user_message_id,
        )

    category_filter = session_mode if session_mode != "general" else None

    if _is_resume_query(user_input):
        doc_type_filter = ["resume"]  # strictly restrict to resume chunks
    else:
        doc_type_filter = ["general", "technical_report"]  # exclude resume from ordinary Q&A

    hits = knowledge_query(user_input, top_k=RAG_TOP_K, category_filter=category_filter,
                            doc_type_filter=doc_type_filter)
    if not hits and _is_resume_query(user_input):
        hits = knowledge_query(user_input, top_k=RAG_TOP_K, category_filter=category_filter,
                                doc_type_filter=None)

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
        prompt = (
            f"The context below is real, stored knowledge retrieved from "
            f"your local knowledge base -- it is not something you need "
            f"to be skeptical of. Multiple relevant pieces of context "
            f"were found; synthesize them into a clear answer. Do not "
            f"claim a term or topic is unfamiliar, unrecognized, or "
            f"outside your knowledge if the context below describes it "
            f"-- the context takes priority over your own training "
            f"knowledge on this topic.\n\n"
            f"Context:\n{context}\n\nQuestion: {user_input}"
        )
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
    messages = [{"role": "system", "content": _build_system_prompt(session_id)}, *recent,
                {"role": "user", "content": prompt}]

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


def route_query(session_id: str, user_input: str, attachment: str | None = None) -> RouteResult:
    prepared = _prepare_response(session_id, user_input, attachment=attachment)
    if prepared.static_answer is not None:
        answer = prepared.static_answer
    else:
        answer = chat(prepared.messages, temperature=prepared.answer_temperature)
    assistant_message_id = memory.add_message(session_id, "assistant", answer)

    return RouteResult(
        answer=answer, mode=prepared.mode, sources=prepared.sources,
        filed_elsewhere=prepared.filed_elsewhere,
        user_message_id=prepared.user_message_id,
        assistant_message_id=assistant_message_id,
    )


def route_query_stream(session_id: str, user_input: str, attachment: str | None = None):
    """
    Generator version: yields small text chunks as they're generated,
    for progressive display in the UI. The full answer is still saved
    to memory once streaming completes, same as the non-streaming path.
    Yields dicts: {"type": "chunk", "text": ...} while streaming, then
    one final {"type": "done", "mode":..., "sources":..., ...} with
    everything the frontend needs once the answer is complete.
    """
    prepared = _prepare_response(session_id, user_input, attachment=attachment)
    full_answer = ""

    if prepared.static_answer is not None:
        full_answer = prepared.static_answer
        yield {"type": "chunk", "text": full_answer}
    else:
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