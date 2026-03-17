from __future__ import annotations

from openai import OpenAI

from app.config import get_settings


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM client is missing required configuration."""


class OpenAICompatibleClient:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.llm_api_key:
            raise LLMConfigurationError(
                "Missing API key. Set one of LLM_API_KEY, GROQ_API_KEY, GROK_API_KEY, or OPENAI_API_KEY."
            )

        self.classifier_model = settings.classifier_model
        self.generation_model = settings.generation_model
        self._client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

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


def build_client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient()