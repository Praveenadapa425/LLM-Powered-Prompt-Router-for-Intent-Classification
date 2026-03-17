# LLM-Powered Prompt Router for Intent Classification

This project is a Python service that classifies a user request, routes it to a specialized AI persona, and returns a focused response. It uses a two-step LLM workflow: a lightweight classification call first, then a second generation call using a domain-specific system prompt. The implementation targets the assignment requirements and is configured to work with a Grok API key through xAI's OpenAI-compatible API.

## Features

- Four expert personas: code, data, writing, and career
- Structured `classify_intent` JSON output with intent and confidence
- Safe fallback to `unclear` on malformed classifier output
- Confidence threshold support
- Manual routing override with prefixes like `@code`
- JSON Lines logging to `route_log.jsonl`
- FastAPI endpoint for local use and evaluation
- Docker and docker-compose support
- Unit tests covering routing and fallback behavior

## Project Structure

```text
app/
  config.py
  llm_client.py
  logger.py
  main.py
  prompts.py
  router.py
  service.py
tests/
  test_router.py
Dockerfile
docker-compose.yml
requirements.txt
.env.example
route_log.jsonl
```

## How It Works

1. `classify_intent(message)` sends the user message to the classifier prompt and expects a JSON response.
2. The JSON is parsed and normalized into this schema:

```json
{
  "intent": "string",
  "confidence": 0.0
}
```

3. `route_and_respond(message, intent_result)` maps the returned intent to an expert system prompt.
4. If the intent is `unclear`, or confidence falls below `CONFIDENCE_THRESHOLD`, the service asks a clarifying question.
5. Each request is appended to `route_log.jsonl` as one valid JSON object per line.

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and set your Grok API key:

```env
GROK_API_KEY=your_real_key
GROK_BASE_URL=https://api.x.ai/v1
CLASSIFIER_MODEL=grok-3-mini
GENERATION_MODEL=grok-3-mini
CONFIDENCE_THRESHOLD=0.70
ROUTE_LOG_FILE=route_log.jsonl
```

## Run the API

```powershell
uvicorn app.main:app --reload
```

The service starts at `http://127.0.0.1:8000`.

### Health Check

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### Route a Message

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/route `
  -ContentType 'application/json' `
  -Body '{"message":"how do i sort a list of objects in python?"}'
```

Example response:

```json
{
  "intent": "code",
  "confidence": 0.93,
  "user_message": "how do i sort a list of objects in python?",
  "final_response": "..."
}
```

## Run Tests

```powershell
pytest
```

## Sample Test Messages

Use these to populate or validate `route_log.jsonl`:

1. how do i sort a list of objects in python?
2. explain this sql query for me
3. This paragraph sounds awkward, can you help me fix it?
4. I'm preparing for a job interview, any tips?
5. what's the average of these numbers: 12, 45, 23, 67, 34
6. Help me make this better.
7. I need to write a function that takes a user id and returns their profile, but also i need help with my resume.
8. hey
9. Can you write me a poem about clouds?
10. Rewrite this sentence to be more professional.
11. I'm not sure what to do with my career.
12. what is a pivot table
13. fxi thsi bug pls: for i in range(10) print(i)
14. How do I structure a cover letter?
15. My boss says my writing is too verbose.

## Docker

Build and run with Docker Compose:

```powershell
docker compose up --build
```

Before running Docker, create a local `.env` file based on `.env.example`.

## Design Notes

- Prompts are stored centrally in `app/prompts.py`, not hardcoded in business logic.
- The classifier is instructed to return JSON only.
- Malformed classifier output is handled safely and defaults to `unclear` with `0.0` confidence.
- Routing never guesses when the request is unsupported or ambiguous.
- Manual overrides such as `@code fix this bug` bypass classification and route directly.

## Submission Checklist

- Application code included
- Dockerfile included
- docker-compose.yml included
- README included
- .env.example included
- route_log.jsonl included"# LLM-Powered-Prompt-Router-for-Intent-Classification" 
