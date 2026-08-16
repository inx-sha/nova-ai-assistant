"""
Isolated cloud LLM synthesis module for NOVA.

Provides a single function, `synthesize_answer`, that uses a cloud provider (Google Gemini Flash)
to synthesize web search snippets into a clean, factual knowledge chunk. All provider-specific
API logic is isolated in this module.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("nova.cloud_llm")

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

# Provider configuration (isolated within this module)
DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"


def synthesize_answer(query: str, search_snippets: list[str]) -> str | None:
    """
    Synthesizes search snippets into a clean, factual answer using Google Gemini Flash.

    Returns the synthesized text chunk on success, or None if:
    - GEMINI_API_KEY is missing or empty
    - search_snippets is empty
    - The API call fails or encounters an error
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.info("GEMINI_API_KEY not configured; skipping cloud synthesis.")
        return None

    if not query.strip() or not search_snippets:
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

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
            "- Do not include meta-commentary, introductory filler, or mentions of 'snippets' or search results.\n"
            "- Present only verified, objective facts."
        )

        model_name = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        if not response or not response.text:
            return None

        synthesized = response.text.strip()
        return synthesized if synthesized else None

    except Exception as e:
        logger.warning(f"Cloud LLM synthesis failed for query '{query}': {e}")
        return None
