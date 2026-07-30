"""
Internet search + fetch + verify pipeline.

Design principle (from the project's Knowledge Verification rules):
never trust a single source. We search multiple results, fetch several
pages, and ask the LLM to cross-check consistency across them before
treating anything as "learned" -- this is what separates it from just
letting the LLM answer from unverified pretrained knowledge.
"""
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
    """
    Searches the web and asks the LLM to produce a verified summary from
    multiple search result snippets -- cross-checking consistency across
    sources rather than trusting any single one. Returns None if nothing
    usable was found.
    """
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

    confidence = 0.6
    summary = response
    if "CONFIDENCE:" in response:
        summary, _, conf_part = response.partition("CONFIDENCE:")
        summary = summary.strip()
        try:
            confidence = max(0.0, min(1.0, float(conf_part.strip().split()[0])))
        except (ValueError, IndexError):
            pass

    return ResearchOutcome(
        summary=summary,
        confidence=confidence,
        sources=[r.url for r in usable],
    )