"""
Isolated cloud LLM synthesis module for NOVA.

Provides a single function, `synthesize_answer`, that uses a cloud provider (Anthropic Claude Haiku)
to synthesize web search snippets into a clean, factual knowledge chunk. All provider-specific
API logic is isolated in this module.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("nova.cloud_llm")

# Provider configuration (isolated within this module)
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")


def synthesize_answer(query: str, search_snippets: list[str]) -> str | None:
    """
    Synthesizes search snippets into a clean, factual answer using Anthropic Claude Haiku.

    Returns the synthesized text chunk on success, or None if:
    - ANTHROPIC_API_KEY is missing or empty
    - search_snippets is empty
    - The API call fails or encounters an error
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        logger.info("ANTHROPIC_API_KEY not configured; skipping cloud synthesis.")
        return None

    if not query.strip() or not search_snippets:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        snippets_block = "\n\n---\n\n".join(
            f"Snippet {i+1}:\n{s.strip()}"
            for i, s in enumerate(search_snippets)
            if s.strip()
        )

        if not snippets_block.strip():
            return None

        prompt = (
            "You are synthesizing web search results into a clean, factual knowledge entry "
            "for an offline knowledge base. Answer the user's question accurately, concisely, "
            "and factually using the provided search snippets.\n\n"
            f"Question: {query}\n\n"
            f"Search Results:\n{snippets_block}\n\n"
            "Instructions:\n"
            "- Provide a clear, self-contained, and factual explanation (2-5 paragraphs).\n"
            "- Do not include meta-commentary, introductory filler, or mentions of 'snippets'.\n"
            "- Present only verified, objective facts."
        )

        message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        if not message.content:
            return None

        result_blocks = []
        for block in message.content:
            if hasattr(block, "text") and block.text:
                result_blocks.append(block.text)

        synthesized = "\n".join(result_blocks).strip()
        return synthesized if synthesized else None

    except Exception as e:
        logger.warning(f"Cloud LLM synthesis failed for query '{query}': {e}")
        return None
