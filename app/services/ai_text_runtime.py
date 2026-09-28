from __future__ import annotations

import os
import re
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

T = TypeVar("T", bound=BaseModel)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_GEMINI_HEALTH_CACHE: dict[str, tuple[bool, int | None, str]] = {}
_GEMINI_HEALTH_LOCK = threading.Lock()


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

def _gemini_error_details(exc: Exception) -> tuple[int | None, str, str]:
    """Extract safe HTTP/API status details from a Google GenAI exception."""
    response_json = getattr(exc, "response_json", None) or getattr(exc, "details", None)
    if not isinstance(response_json, dict):
        response_json = {}
    error_data = response_json.get("error", response_json)
    if not isinstance(error_data, dict):
        error_data = {}

    http_code = _status_code(exc)
    if http_code is None:
        candidate = error_data.get("code")
        if isinstance(candidate, int):
            http_code = candidate

    api_status = error_data.get("status") or getattr(exc, "status", None) or "unknown"
    message = error_data.get("message") or getattr(exc, "message", None) or str(exc) or type(exc).__name__
    return http_code, str(api_status), _redact_gemini_message(str(message))


def _redact_gemini_message(message: str) -> str:
    message = re.sub(r"(?i)authorization\s*:\s*[^,;]+", "Authorization: [redacted]", message)
    message = re.sub(r"(?i)(bearer|AIza|sk-or-v1|sk-proj)-?[A-Za-z0-9_\-\.]+", "[redacted]", message)
    secret = os.getenv("GEMINI_API_KEY", "").strip()
    if secret:
        message = message.replace(secret, "[redacted]")
    return message[:500]


def _safe_error_message(exc: Exception) -> str:
    """Extract a useful provider error without logging credentials or headers."""
    if isinstance(exc, _OpenRouterResponseError):
        return str(exc)
    code = _status_code(exc)
    return f"HTTP {code}" if code is not None else type(exc).__name__


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
        result = parsed
    elif parsed is not None:
        result = response_model.model_validate(parsed)
    else:
        raw = getattr(response, "text", None)
        if not raw:
            raise ValueError("Gemini returned an empty response.")
        result = response_model.model_validate_json(raw)

    panels = getattr(result, "panels", None)
    if isinstance(panels, list):
        panel_numbers = [getattr(panel, "panel_number", None) for panel in panels]
        if len(panels) != 5 or panel_numbers != [1, 2, 3, 4, 5]:
            raise ValueError("Gemini response must contain exactly five ordered panels.")
    return result


def clear_gemini_health_cache() -> None:
    """Clear cached Gemini availability results, primarily for tests or key changes."""
    with _GEMINI_HEALTH_LOCK:
        _GEMINI_HEALTH_CACHE.clear()


def _check_gemini_model(model: str) -> tuple[bool, int | None, str]:
    client = None
    try:
        client = _gemini_client()
        client.models.generate_content(
            model=model,
            contents="OK",
            config=types.GenerateContentConfig(max_output_tokens=1),
        )
        return True, None, ""
    except Exception as exc:
        return False, _status_code(exc), type(exc).__name__
    finally:
        if client is not None:
            client.close()


def _healthy_gemini_models(models: list[str]) -> list[str]:
    """Probe uncached models once, concurrently, and return healthy models in priority order."""
    with _GEMINI_HEALTH_LOCK:
        uncached = [model for model in models if model not in _GEMINI_HEALTH_CACHE]
        newly_checked = set(uncached)
        if uncached:
            print("[AI] Gemini health check started.")
            with ThreadPoolExecutor(max_workers=min(5, len(uncached))) as executor:
                futures = {
                    executor.submit(_check_gemini_model, model): model
                    for model in uncached
                }
                for future in as_completed(futures):
                    model = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = (False, _status_code(exc), type(exc).__name__)
                    _GEMINI_HEALTH_CACHE[model] = result

        for model in models:
            available, code, reason = _GEMINI_HEALTH_CACHE[model]
            if model in newly_checked:
                if available:
                    print(f"[AI] Gemini model: {model} → AVAILABLE")
                else:
                    detail = f"HTTP {code}" if code is not None else reason
                    print(f"[AI] Gemini model: {model} → UNAVAILABLE ({detail})")
            else:
                print(f"[AI] Gemini model: {model} → using cached availability result")

        return [model for model in models if _GEMINI_HEALTH_CACHE[model][0]]


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
    result = _validate_openrouter_response(raw, response_model)
    panel_count = len(getattr(result, "panels", []))
    print(f"[AI] OpenRouter response validated successfully: {panel_count} panels")
    return result


def generate_structured_with_fallback(
    *,
    prompt: str,
    response_model: type[T],
    gemini_models: tuple[str, ...],
    local_fallback: Callable[[], T],
    gemini_env_name: str = "GEMINI_OUTLINE_MODEL",
) -> TextGenerationResult:
    """Generate validated structured text through OpenRouter, Gemini, then local fallback."""
    model = os.getenv("OPENROUTER_TEXT_MODEL", "openrouter/free").strip()
    if os.getenv("OPENROUTER_API_KEY", "").strip():
        try:
            print("[AI] Provider: OpenRouter")
            print(f"[AI] OpenRouter model: {model}")
            print("[AI] OpenRouter request started.")
            result = _openrouter_structured(prompt, response_model)
            print("[AI] Final text provider: OpenRouter")
            return TextGenerationResult(result, "openrouter")
        except Exception as exc:
            print(f"[AI] OpenRouter unavailable: {_safe_error_message(exc)}")
    else:
        print("[AI] OpenRouter unavailable: OPENROUTER_API_KEY is not configured.")

    print("[AI] Falling back to Gemini.")
    configured_models = _candidate_models(gemini_env_name, gemini_models)
    if os.getenv("GEMINI_API_KEY", "").strip():
        healthy_models = _healthy_gemini_models(configured_models)
        if healthy_models:
            print("[AI] Provider: Gemini")
        for gemini_model in healthy_models:
            client = None
            try:
                client = _gemini_client()
                response = client.models.generate_content(
                    model=gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_model,
                    ),
                )
                result = _parse_gemini_response(response, response_model)
                print(f"[AI] Gemini generation successful: {gemini_model}")
                print("[AI] Final text provider: Gemini")
                return TextGenerationResult(result, "gemini")
            except Exception as exc:
                code = _status_code(exc)
                reason = f"HTTP {code}" if code is not None else type(exc).__name__
                print(f"[AI] Gemini generation failed: {gemini_model} ({reason})")
            finally:
                if client is not None:
                    client.close()
    else:
        print("[AI] Gemini unavailable: GEMINI_API_KEY is not configured.")

    print("[AI] Gemini unavailable.")

    print("[AI] Using local text fallback.")
    result = local_fallback()
    print("[AI] Final text provider: local fallback")
    return TextGenerationResult(result, "local-fallback")