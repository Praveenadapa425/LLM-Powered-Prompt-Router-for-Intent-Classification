from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.llm_client import LLMConfigurationError
from app.service import process_message


app = FastAPI(title="LLM-Powered Prompt Router", version="1.0.0")


class RouteRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to classify and route")


class RouteResponse(BaseModel):
    intent: str
    confidence: float
    user_message: str
    final_response: str
    manual_override: bool | None = None


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
def route_message(request: RouteRequest) -> RouteResponse:
    try:
        result = process_message(request.message)
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=500, detail=f"Unexpected routing error: {exc}") from exc

    return RouteResponse(**result)