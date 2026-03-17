from __future__ import annotations

from typing import Any, Dict

from app.llm_client import build_client
from app.logger import append_route_log
from app.router import classify_intent, route_and_respond


def process_message(message: str) -> Dict[str, Any]:
    client = build_client()
    intent_result = classify_intent(message, client)
    final_response = route_and_respond(message, intent_result, client)

    payload = {
        "intent": intent_result["intent"],
        "confidence": float(intent_result["confidence"]),
        "user_message": message,
        "final_response": final_response,
    }
    if "manual_override" in intent_result:
        payload["manual_override"] = bool(intent_result["manual_override"])

    append_route_log(payload)
    return payload