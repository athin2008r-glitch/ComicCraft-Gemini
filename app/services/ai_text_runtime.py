from __future__ import annotations

import os
import re
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

T = TypeVar("T", bound=BaseModel)
RETRYABLE_CODES = {429, 500, 502, 503, 504}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class TextGenerationResult:
    value: BaseModel
    provider: str


class _OpenRouterResponseError(ValueError):
    """Safe, user-independent error for an invalid OpenRouter response."""


class _OpenRouterJsonError(_OpenRouterResponseError):
    pass


class _OpenRouterSchemaError(_OpenRouterResponseError):
    pass


def _status_code(exc: Exception) -> int | None:
    for attribute in ("code", "status_code", "status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    request = getattr(exc, "request", None)
    response = getattr(request, "response", None) if request else None
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _safe_error_message(exc: Exception) -> str:
    """Extract a useful provider error without logging credentials or headers."""
    body = getattr(exc, "body", None)
    if not isinstance(body, (dict, list)):
        response = getattr(exc, "response", None)
        try:
            body = response.json() if response is not None else None
        except Exception:
            body = None

    message = None
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            message = error.get("message") or error.get("detail") or error.get("code")
        elif isinstance(error, str):
            message = error
    elif isinstance(body, list):
        message = "; ".join(str(item) for item in body[:2])

    if not message:
        message = str(exc) or type(exc).__name__

    message = re.sub(r"(?i)authorization\s*:\s*[^,;]+", "Authorization: [redacted]", str(message))
    message = re.sub(r"(?i)(bearer|sk-or-v1|sk-proj)-?[A-Za-z0-9_\-\.]+", "[redacted]", message)
    secret = os.getenv("OPENROUTER_API_KEY", "").strip()
    if secret:
        message = message.replace(secret, "[redacted]")
    return message[:500]


def _extract_json_object(raw: str) -> object:
    """Parse a JSON object, allowing only surrounding markdown fences/text."""
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE).strip()

    start = candidate.find("{")
    if start < 0:
        raise _OpenRouterJsonError("OpenRouter JSON parsing failed: no JSON object found.")

    try:
        value, _ = json.JSONDecoder().raw_decode(candidate[start:])
    except json.JSONDecodeError as exc:
        raise _OpenRouterJsonError(
            f"OpenRouter JSON parsing failed: invalid JSON at line {exc.lineno} column {exc.colno}."
        ) from exc
    if not isinstance(value, dict):
        raise _OpenRouterJsonError("OpenRouter JSON parsing failed: expected a JSON object.")
    return value


def _validate_openrouter_response(raw: str, response_model: type[T]) -> T:
    value = _extract_json_object(raw)
    try:
        result = response_model.model_validate(value)
    except ValidationError as exc:
        first_error = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first_error.get("loc", ())) or "response"
        message = first_error.get("msg", "invalid response")
        raise _OpenRouterSchemaError(
            f"OpenRouter schema validation failed: {location}: {message}."
        ) from exc

    panels = getattr(result, "panels", None)
    if isinstance(panels, list):
        panel_numbers = [getattr(panel, "panel_number", None) for panel in panels]
        if len(panels) != 5 or panel_numbers != [1, 2, 3, 4, 5]:
            raise _OpenRouterSchemaError(
                "OpenRouter schema validation failed: panels must contain exactly five "
                "ordered panels numbered 1 through 5."
            )
    return result


def _retry_delay(attempt: int) -> float:
    base = max(0.0, float(os.getenv("GEMINI_RETRY_BACKOFF", "1.5")))
    return base * attempt


def _gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    return genai.Client(api_key=api_key)


def _candidate_models(configured_name: str, defaults: tuple[str, ...]) -> list[str]:
    configured = os.getenv(configured_name, "").strip()
    return list(dict.fromkeys(([configured] if configured else []) + list(defaults)))


def _parse_gemini_response(response: object, response_model: type[T]) -> T:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, response_model):
        return parsed
    if parsed is not None:
        return response_model.model_validate(parsed)
    raw = getattr(response, "text", None)
    if not raw:
        raise ValueError("Gemini returned an empty response.")
    return response_model.model_validate_json(raw)


def _openrouter_structured(
    prompt: str,
    response_model: type[T],
) -> T:
    from openai import OpenAI as OpenRouterClient

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")

    model = os.getenv("OPENROUTER_TEXT_MODEL", "openrouter/free").strip()
    client = OpenRouterClient(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    schema = json.dumps(response_model.model_json_schema(), separators=(",", ":"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return exactly one complete JSON object matching this JSON Schema. "
                    "Do not return safety labels, commentary, a single item, or markdown fences. "
                    f"The object must include every required field and complete arrays. Schema: {schema}"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content if response.choices else None
    if not raw:
        raise ValueError("OpenRouter returned an empty response.")
    return _validate_openrouter_response(raw, response_model)


def generate_structured_with_fallback(
    *,
    prompt: str,
    response_model: type[T],
    gemini_models: tuple[str, ...],
    local_fallback: Callable[[], T],
    gemini_env_name: str = "GEMINI_OUTLINE_MODEL",
) -> TextGenerationResult:
    """Generate validated structured text through Gemini, OpenRouter, then local fallback."""
    configured = os.getenv(gemini_env_name, "").strip()
    configured_models = list(dict.fromkeys(([configured] if configured else []) + list(gemini_models)))
    if not configured_models:
        configured_models = list(gemini_models)
    attempts = max(1, int(os.getenv("GEMINI_RETRIES_PER_MODEL", "2")))

    if os.getenv("GEMINI_API_KEY", "").strip():
        for model in configured_models:
            print("[AI] Provider: Gemini")
            for attempt in range(1, attempts + 1):
                try:
                    response = _gemini_client().models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=response_model,
                        ),
                    )
                    result = _parse_gemini_response(response, response_model)
                    print(f"[AI] Gemini generation successful: {model}")
                    return TextGenerationResult(result, "gemini")
                except Exception as exc:
                    code = _status_code(exc)
                    print(f"[AI] Gemini model failed: {model} HTTP {code or '?'}")
                    if code not in RETRYABLE_CODES or attempt >= attempts:
                        break
                    time.sleep(_retry_delay(attempt))
        print("[AI] Gemini unavailable.")
    else:
        print("[AI] Gemini skipped: GEMINI_API_KEY is not configured.")

    if os.getenv("OPENROUTER_API_KEY", "").strip():
        print("[AI] Falling back to OpenRouter.")
        try:
            model = os.getenv("OPENROUTER_TEXT_MODEL", "openrouter/free").strip()
            print("[AI] Provider: OpenRouter")
            print(f"[AI] OpenRouter model: {model}")
            result = _openrouter_structured(prompt, response_model)
            return TextGenerationResult(result, "openrouter")
        except Exception as exc:
            code = _status_code(exc)
            print(f"[AI] OpenRouter model failed: {model} HTTP {code or '?'}")
            print(f"[AI] OpenRouter error: {_safe_error_message(exc)}")
            if isinstance(exc, _OpenRouterJsonError):
                print("[AI] OpenRouter JSON parsing failure.")
            elif isinstance(exc, _OpenRouterSchemaError):
                print("[AI] OpenRouter schema validation failure.")
            print("[AI] OpenRouter unavailable.")
    else:
        print("[AI] OpenRouter skipped: OPENROUTER_API_KEY is not configured.")

    print("[AI] Using local text fallback.")
    return TextGenerationResult(local_fallback(), "local-fallback")