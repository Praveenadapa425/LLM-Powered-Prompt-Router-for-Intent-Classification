from __future__ import annotations

import json

from app.router import build_clarification_question, classify_intent, route_and_respond


class FakeClient:
    classifier_model = "fake-classifier"
    generation_model = "fake-generation"

    def __init__(self, classifier_response: str = '{"intent": "code", "confidence": 0.95}', generation_response: str = "Generated response") -> None:
        self.classifier_response = classifier_response
        self.generation_response = generation_response
        self.calls = []

    def complete(self, *, system_prompt: str, user_message: str, model: str, temperature: float = 0.2) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "model": model,
                "temperature": temperature,
            }
        )
        if model == self.classifier_model:
            return self.classifier_response
        return self.generation_response


def test_classify_intent_parses_valid_json() -> None:
    client = FakeClient(classifier_response='{"intent": "data", "confidence": 0.91}')
    result = classify_intent("what is the average of 1, 2, 3", client)

    assert result["intent"] == "data"
    assert result["confidence"] == 0.91
    assert result["manual_override"] is False


def test_classify_intent_defaults_on_invalid_json() -> None:
    client = FakeClient(classifier_response="not-json")
    result = classify_intent("help", client)

    assert result["intent"] == "unclear"
    assert result["confidence"] == 0.0


def test_manual_override_skips_classifier_call() -> None:
    client = FakeClient()
    result = classify_intent("@code fix this python bug", client)

    assert result["intent"] == "code"
    assert result["confidence"] == 1.0
    assert client.calls == []


def test_route_and_respond_asks_for_clarification_when_unclear() -> None:
    client = FakeClient()
    final_response = route_and_respond("hey", {"intent": "unclear", "confidence": 0.0}, client)

    assert final_response == build_clarification_question()


def test_route_and_respond_uses_generation_prompt() -> None:
    client = FakeClient(generation_response="Use sorted(items, key=lambda row: row['name'])")
    final_response = route_and_respond(
        "sort a list of objects in python",
        {"intent": "code", "confidence": 0.99, "cleaned_message": "sort a list of objects in python"},
        client,
    )

    assert final_response.startswith("Use sorted")
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == client.generation_model


def test_route_and_respond_treats_low_confidence_as_unclear(monkeypatch) -> None:
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.80")
    from app.config import get_settings

    get_settings.cache_clear()
    client = FakeClient()
    final_response = route_and_respond("what is pivot table", {"intent": "data", "confidence": 0.5}, client)

    assert final_response == build_clarification_question()
    get_settings.cache_clear()


def test_invalid_intent_is_normalized_to_unclear() -> None:
    client = FakeClient(classifier_response=json.dumps({"intent": "poetry", "confidence": 0.88}))
    result = classify_intent("write me a poem", client)

    assert result["intent"] == "unclear"


def test_guardrail_routes_sql_to_code() -> None:
    client = FakeClient(classifier_response='{"intent": "writing", "confidence": 0.8}')
    result = classify_intent("explain this sql query for me", client)

    assert result["intent"] == "code"
    assert result["confidence"] >= 0.9


def test_guardrail_routes_creative_requests_to_unclear() -> None:
    client = FakeClient(classifier_response='{"intent": "writing", "confidence": 0.9}')
    result = classify_intent("Can you write me a poem about clouds?", client)

    assert result["intent"] == "unclear"
    assert result["confidence"] == 0.0


def test_guardrail_routes_verbose_feedback_to_writing() -> None:
    client = FakeClient(classifier_response='{"intent": "career", "confidence": 0.8}')
    result = classify_intent("My boss says my writing is too verbose.", client)

    assert result["intent"] == "writing"
    assert result["confidence"] >= 0.9


def test_guardrail_routes_mixed_intent_to_unclear() -> None:
    client = FakeClient(classifier_response='{"intent": "career", "confidence": 0.8}')
    result = classify_intent(
        "I need to write a function that takes a user id and returns their profile, and I also need help with my resume.",
        client,
    )

    assert result["intent"] == "unclear"
    assert result["confidence"] == 0.0