import logging
import re
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS  # type: ignore
except ImportError:
    from duckduckgo_search import DDGS  # type: ignore

from core.llm import LLMError, chat

logger = logging.getLogger("nova.internet")

SEARCH_MAX_RESULTS = 5
FETCH_TOP_N = 3
FETCH_TIMEOUT = 15.0
MAX_PAGE_CHARS = 6000  # keep fetched text within a reasonable prompt size


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class ResearchOutcome:
    summary: str
    confidence: float
    sources: list[str]


def search_web(query: str, max_results: int = SEARCH_MAX_RESULTS) -> list[SearchResult]:
    results: list[SearchResult] = []
    try:
        with DDGS() as ddgs:
            raw_results = ddgs.text(query, max_results=max_results)
            if raw_results:
                for r in raw_results:
                    title = r.get("title") or ""
                    url = r.get("href") or r.get("url") or r.get("link") or ""
                    snippet = r.get("body") or r.get("snippet") or r.get("description") or ""
                    if snippet.strip() or url.strip():
                        results.append(SearchResult(
                            title=title.strip(),
                            url=url.strip(),
                            snippet=snippet.strip(),
                        ))
    except Exception as e:
        logger.warning(f"Web search failed for query '{query}': {e}")
        return []
    return results


def fetch_page_text(url: str) -> str | None:
    """Returns cleaned readable text from a page, or None if fetch fails."""
    try:
        resp = httpx.get(
            url,
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = " ".join(soup.get_text(separator=" ").split())
        return text[:MAX_PAGE_CHARS] if text else None
    except Exception as e:
        logger.debug(f"Failed to fetch page text for {url}: {e}")
        return None


def research_topic(query: str) -> ResearchOutcome | None:
    try:
        results = search_web(query)
    except Exception as e:
        logger.warning(f"search_web raised unexpected error for '{query}': {e}")
        return None

    if not results:
        return None

    usable = [r for r in results if r.snippet.strip() and r.url.strip()]
    if not usable:
        return None

    sources_block = "\n\n---\n\n".join(
        f"SOURCE: {r.url}\nTITLE: {r.title}\n{r.snippet}" for r in usable
    )

    verification_prompt = (
        "You are verifying information across multiple web search results "
        "before it gets stored as trusted knowledge. Below are snippets "
        f"from {len(usable)} different sources about this question:\n\n"
        f"Question: {query}\n\n{sources_block}\n\n"
        "Instructions:\n"
        "1. Write a concise, factual summary (3-6 sentences) answering the "
        "question, using ONLY information found in these snippets.\n"
        "2. On a new line, write 'CONFIDENCE: X' where X is a number from "
        "0.0 to 1.0 -- how well the sources agree with each other. Use 0.9+ "
        "only if sources clearly agree; use 0.5-0.7 if sources partially "
        "agree or only one source covers it well; use below 0.5 if sources "
        "conflict or are vague.\n"
        "Do not include anything else in your response."
    )

    try:
        response = chat([
            {"role": "user", "content": verification_prompt}
        ])
    except (LLMError, Exception) as e:
        logger.error(f"LLM verification failed during research for '{query}': {e}")
        return None

    if not response or not response.strip():
        return None

    confidence = 0.6
    clean_response = response.strip()

    # Extract confidence score (e.g. CONFIDENCE: 0.85, **Confidence**: 0.9, **Confidence:** 0.95)
    conf_pattern = r"(?:\*\*|\*|#)*\s*(?:CONF\w*|CONFIDENCE)\s*(?:\*\*|\*|#)*\s*[:=\-]?\s*(?:\*\*|\*|#)*\s*(\d*\.?\d+)(?:\*\*|\*|#)*"
    conf_match = re.search(conf_pattern, clean_response, re.IGNORECASE)
    if conf_match:
        try:
            val = float(conf_match.group(1))
            confidence = max(0.0, min(1.0, val))
        except ValueError:
            pass

        # Remove the matched confidence snippet from the summary text
        summary = (clean_response[:conf_match.start()] + clean_response[conf_match.end():]).strip()
    else:
        summary = clean_response

    # Clean leading labels (e.g. "**SUMMARY:** ", "Summary: ", etc.) without stripping actual sentences
    summary = re.sub(r"^(?:\*\*|\*|#)*\s*(?:SUMMARY|ANSWER)\s*(?:\*\*|\*|#)*\s*[:\-]\s*(?:\*\*|\*|#)*\s*", "", summary, flags=re.IGNORECASE).strip()
    summary = re.sub(r"^(?:\*\*|\*|#)+\s*(?:SUMMARY|ANSWER)\s*(?:\*\*|\*|#)+\s*", "", summary, flags=re.IGNORECASE).strip()
    summary = re.sub(r"^\*+\s*", "", summary).strip()
    summary = re.sub(r"\*+\s*$", "", summary).strip()

    if not summary:
        return None

    return ResearchOutcome(
        summary=summary,
        confidence=confidence,
        sources=[r.url for r in usable],
    )


def verify_claim(topic: str, claimed_correct_info: str) -> tuple[str, str]:
    outcome = research_topic(topic)
    if outcome is None:
        return "unverified", "Could not research this online to verify (no internet or no results)."

    verification_prompt = (
        f"A user claims the following about '{topic}':\n\n"
        f"USER'S CLAIM: {claimed_correct_info}\n\n"
        f"INDEPENDENT RESEARCH SUMMARY: {outcome.summary}\n\n"
        "Does the independent research SUPPORT, CONTRADICT, or NOT ADDRESS "
        "the user's claim? Respond with exactly one word first "
        "(SUPPORT, CONTRADICT, or UNCLEAR), then a colon, then a one-"
        "sentence explanation. Example: 'SUPPORT: both describe the same mechanism.'"
    )

    try:
        response = chat([
            {"role": "user", "content": verification_prompt}
        ])
    except (LLMError, Exception) as e:
        logger.error(f"LLM verification failed during claim verification for '{topic}': {e}")
        return "unverified", f"Verification query failed: {e}"

    clean_resp = response.strip()
    if not clean_resp:
        return "unverified", "Empty verification response from model."

    # Parse verdict robustly
    first_part = clean_resp.split(":")[0] if ":" in clean_resp else clean_resp.split("\n")[0]
    verdict_token = re.sub(r"[^\w\s]", "", first_part).strip().upper()
    tokens = verdict_token.split()

    if any(t in ("SUPPORT", "SUPPORTS", "AGREE", "AGREES", "CONFIRM", "CONFIRMS") for t in tokens):
        return "verified", clean_resp
    elif any(t in ("CONTRADICT", "CONTRADICTS", "DISPUTE", "DISPUTES", "REFUTE", "REFUTES", "FALSE") for t in tokens):
        return "disputed", clean_resp
    else:
        return "unverified", clean_resp