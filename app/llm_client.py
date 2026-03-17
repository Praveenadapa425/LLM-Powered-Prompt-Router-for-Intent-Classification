from __future__ import annotations

from typing import Any, Dict, List

from openai import OpenAI

from app.config import get_settings


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM client is missing required configuration."""


class GrokClient:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.grok_api_key:
            raise LLMConfigurationError(
                "Missing GROK_API_KEY. Set it in your environment or .env file."
            )

        self.classifier_model = settings.classifier_model
        self.generation_model = settings.generation_model
        self._client = OpenAI(api_key=settings.grok_api_key, base_url=settings.grok_base_url)

    def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        temperature: float = 0.2,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""


def build_client() -> GrokClient:
    return GrokClient()