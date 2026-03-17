from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    grok_api_key: str = os.getenv("GROK_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    grok_base_url: str = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
    classifier_model: str = os.getenv("CLASSIFIER_MODEL", "grok-3-mini")
    generation_model: str = os.getenv("GENERATION_MODEL", "grok-3-mini")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
    log_file: str = os.getenv("ROUTE_LOG_FILE", "route_log.jsonl")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()