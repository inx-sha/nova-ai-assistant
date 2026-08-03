from __future__ import annotations

from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

from core.llm import chat

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
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
            ))
    return results


def fetch_page_text(url: str) -> str | None:
    """Returns cleaned readable text from a page, or None if fetch fails."""
    try:
        resp = httpx.get(url, timeout=FETCH_TIMEOUT, follow_redirects=True,
                  headers={
                      "User-Agent": (
                          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0.0.0 Safari/537.36"
                      )
                  })
        resp.raise_for_status()
    except httpx.HTTPError:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    return text[:MAX_PAGE_CHARS] if text else None


def research_topic(query: str) -> ResearchOutcome | None:

    results = search_web(query)
    if not results:
        return None

    usable = [r for r in results if r.snippet.strip()]
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

    response = chat([{"role": "user", "content": verification_prompt}])

    import re
    confidence = 0.6
    summary = response

    # Match CONFIDENCE (and common misspellings/variants the model produces)
    # followed by a number, case-insensitive, regardless of ** markdown or colons.
    match = re.search(r"CONF\w*\s*:?\s*\**\s*(\d*\.?\d+)", response, re.IGNORECASE)
    if match:
        summary = response[:match.start()].strip()
        try:
            confidence = max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            pass

    # Strip any leftover markdown bold markers or "SUMMARY:" labels from the front
    summary = re.sub(r"\*+\s*$", "", summary).strip()

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
    response = chat([{"role": "user", "content": verification_prompt}])

    verdict = response.strip().split(":")[0].strip().upper()
    if verdict == "SUPPORT":
        return "verified", response
    elif verdict == "CONTRADICT":
        return "disputed", response
    else:
        return "unverified", response