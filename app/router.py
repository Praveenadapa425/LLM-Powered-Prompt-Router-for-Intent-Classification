from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Protocol, Tuple

from app.config import get_settings
from app.prompts import CLARIFICATION_PROMPT, CLASSIFIER_PROMPT, SUPPORTED_INTENTS, SYSTEM_PROMPTS


class CompletionClient(Protocol):
    classifier_model: str
    generation_model: str

    def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        temperature: float = 0.2,
    ) -> str: ...


def _default_intent() -> Dict[str, Any]:
    return {"intent": "unclear", "confidence": 0.0}


def _extract_json_object(raw_response: str) -> str:
    trimmed = raw_response.strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        return trimmed

    match = re.search(r"\{.*\}", trimmed, re.DOTALL)
    if match:
        return match.group(0)

    return trimmed


def _normalize_intent(payload: Dict[str, Any]) -> Dict[str, Any]:
    intent = str(payload.get("intent", "unclear")).strip().lower()
    if intent not in SUPPORTED_INTENTS:
        intent = "unclear"

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))
    return {"intent": intent, "confidence": confidence}


def _extract_manual_override(message: str) -> Tuple[Optional[str], str]:
    match = re.match(r"^@(code|data|writing|career)\b\s*(.*)$", message.strip(), re.IGNORECASE | re.DOTALL)
    if not match:
        return None, message

    intent = match.group(1).lower()
    cleaned_message = match.group(2).strip() or message.strip()
    return intent, cleaned_message


def classify_intent(message: str, client: CompletionClient) -> Dict[str, Any]:
    manual_intent, cleaned_message = _extract_manual_override(message)
    if manual_intent:
        return {"intent": manual_intent, "confidence": 1.0, "manual_override": True, "cleaned_message": cleaned_message}

    raw_response = client.complete(
        system_prompt=CLASSIFIER_PROMPT,
        user_message=message,
        model=client.classifier_model,
        temperature=0.0,
    )

    try:
        payload = json.loads(_extract_json_object(raw_response))
    except json.JSONDecodeError:
        result = _default_intent()
        result["raw_classifier_output"] = raw_response
        return result

    result = _normalize_intent(payload)
    result["raw_classifier_output"] = raw_response
    result["cleaned_message"] = cleaned_message
    result["manual_override"] = False
    return result


def build_clarification_question() -> str:
    return CLARIFICATION_PROMPT


def route_and_respond(message: str, intent_result: Dict[str, Any], client: CompletionClient) -> str:
    settings = get_settings()
    intent = intent_result.get("intent", "unclear")
    confidence = float(intent_result.get("confidence", 0.0))
    cleaned_message = str(intent_result.get("cleaned_message") or message)

    if intent == "unclear" or confidence < settings.confidence_threshold:
        return build_clarification_question()

    system_prompt = SYSTEM_PROMPTS.get(intent)
    if not system_prompt:
        return build_clarification_question()

    return client.complete(
        system_prompt=system_prompt,
        user_message=cleaned_message,
        model=client.generation_model,
        temperature=0.4,
    )