from __future__ import annotations

"""
Optional LLM Provider Adapter.

Supports enriching narratives via external LLMs (e.g. Gemini API) if configured.
Designed to fail gracefully and return None if API keys are missing or offline.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def is_llm_available() -> bool:
    """Check if GEMINI_API_KEY or OPENAI_API_KEY env vars are present."""
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"))


def generate_llm_enrichment(prompt: str, system_instruction: str = "") -> str | None:
    """
    Generate an enriched narrative via external LLM API if configured.

    Args:
        prompt: User or evidence prompt.
        system_instruction: System prompt.

    Returns:
        Generated string if successful, None if unavailable/offline.
    """
    if not is_llm_available():
        return None

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
            response = model.generate_content(full_prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as exc:
            logger.info(f"Gemini API enrichment skipped: {exc}")

    return None
