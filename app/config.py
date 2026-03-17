from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()


def _resolve_api_key() -> str:
    return (
        os.getenv("LLM_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("GROK_API_KEY")
        or os.getenv("OPENAI_API_KEY", "")
    )


def _resolve_base_url() -> str:
    return (
        os.getenv("LLM_BASE_URL")
        or os.getenv("GROQ_BASE_URL")
        or os.getenv("GROK_BASE_URL")
        or "https://api.x.ai/v1"
    )


def _default_model(base_url: str) -> str:
    if "groq.com" in base_url.lower():
        return "llama-3.1-8b-instant"
    return "grok-3-mini"


@dataclass(frozen=True)
class Settings:
    llm_api_key: str = field(default_factory=_resolve_api_key)
    llm_base_url: str = field(default_factory=_resolve_base_url)
    classifier_model: str = field(
        default_factory=lambda: os.getenv(
            "CLASSIFIER_MODEL", _default_model(_resolve_base_url())
        )
    )
    generation_model: str = field(
        default_factory=lambda: os.getenv(
            "GENERATION_MODEL", _default_model(_resolve_base_url())
        )
    )
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
    log_file: str = os.getenv("ROUTE_LOG_FILE", "route_log.jsonl")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()