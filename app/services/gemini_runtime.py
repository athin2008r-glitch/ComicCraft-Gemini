from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

T = TypeVar("T", bound=BaseModel)

# Errors where trying another model makes sense.
RETRYABLE_CODES = {429, 500, 502, 503, 504}

# 404 means the model cannot be used by the current account.
# 429/5xx can be temporary availability/quota/server problems.
FALLBACK_CODES = {404, 429, 500, 502, 503, 504}


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Create and cache the Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. "
            "Add your Gemini API key to the .env file."
        )

    return genai.Client(api_key=api_key)


def _status_code(exc: Exception) -> int | None:
    """Extract an HTTP/API status code from a Gemini exception."""
    code = getattr(exc, "code", None)

    if code is None:
        code = getattr(exc, "status_code", None)

    if isinstance(code, int):
        return code

    if isinstance(code, str):
        try:
            return int(code)
        except ValueError:
            return None

    return None


def candidate_models(
    env_name: str,
    defaults: tuple[str, ...],
) -> list[str]:
    """
    Put the model configured in .env first, followed by fallbacks.
    Duplicate model names are removed.
    """
    configured = os.getenv(env_name, "").strip()

    models: list[str] = []

    if configured:
        models.append(configured)

    for model in defaults:
        if model not in models:
            models.append(model)

    return models


def _retry_delay(attempt: int) -> float:
    """
    Small exponential backoff.

    Example with default settings:
    attempt 1 -> 1.5 sec
    attempt 2 -> 3.0 sec
    """
    base = max(
        0.0,
        float(os.getenv("GEMINI_RETRY_BACKOFF", "1.5")),
    )

    return base * attempt


def generate_structured(
    *,
    prompt: str,
    response_model: type[T],
    env_name: str,
    defaults: tuple[str, ...],
) -> T:
    """
    Generate structured Gemini output with retries and model fallback.

    The function:
    1. Tries the configured model.
    2. Retries temporary 429/5xx errors.
    3. Moves to the next model if the problem persists.
    4. Skips 404 models immediately.
    5. Validates the final response with Pydantic.
    """

    models = candidate_models(env_name, defaults)

    attempts_per_model = max(
        1,
        int(os.getenv("GEMINI_RETRIES_PER_MODEL", "2")),
    )

    failures: list[str] = []

    print()
    print("=== ComicCraft Gemini Request ===")
    print(f"Models available for fallback: {models}")
    print()

    for model in models:

        print(f"[Gemini] Trying model: {model}")

        for attempt in range(1, attempts_per_model + 1):

            print(
                f"[Gemini] Attempt {attempt}/{attempts_per_model}: "
                f"{model}"
            )

            try:
                response = get_client().models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_model,
                    ),
                )

                # Preferred path when the SDK already parsed the
                # structured response.
                parsed = getattr(response, "parsed", None)

                if isinstance(parsed, response_model):
                    print(f"[Gemini] SUCCESS: {model}")
                    return parsed

                # Some SDK versions may return a dictionary-like
                # parsed object.
                if parsed is not None:
                    result = response_model.model_validate(parsed)

                    print(f"[Gemini] SUCCESS: {model}")

                    return result

                # Fallback: validate raw JSON text.
                raw = getattr(response, "text", None)

                if not raw:
                    raise RuntimeError(
                        f"Gemini returned an empty response from {model}."
                    )

                result = response_model.model_validate_json(raw)

                print(f"[Gemini] SUCCESS: {model}")

                return result

            except errors.APIError as exc:

                code = _status_code(exc)

                message = str(exc)

                failures.append(
                    f"{model} attempt {attempt}: "
                    f"HTTP {code or '?'} - {message}"
                )

                print(
                    f"[Gemini] FAILED: {model} "
                    f"(HTTP {code or '?'})"
                )

                # The account cannot use this model.
                # Don't waste another attempt.
                if code == 404:
                    print(
                        f"[Gemini] Skipping unavailable model: {model}"
                    )
                    break

                # Temporary service/load/quota problem.
                if code in RETRYABLE_CODES:

                    if attempt < attempts_per_model:
                        delay = _retry_delay(attempt)

                        print(
                            f"[Gemini] Temporary failure. "
                            f"Retrying in {delay:.1f}s..."
                        )

                        time.sleep(delay)
                        continue

                    print(
                        f"[Gemini] {model} still unavailable. "
                        f"Trying next model..."
                    )

                    break

                # Unknown API error.
                print(
                    f"[Gemini] Non-retryable API error. "
                    f"Trying next model..."
                )

                break

            except (TimeoutError, ConnectionError, OSError) as exc:

                failures.append(
                    f"{model} attempt {attempt}: "
                    f"{type(exc).__name__}: {exc}"
                )

                print(
                    f"[Gemini] Connection problem: {exc}"
                )

                if attempt < attempts_per_model:

                    delay = _retry_delay(attempt)

                    print(
                        f"[Gemini] Retrying in {delay:.1f}s..."
                    )

                    time.sleep(delay)

                    continue

                break

            except Exception as exc:

                failures.append(
                    f"{model} attempt {attempt}: "
                    f"{type(exc).__name__}: {exc}"
                )

                print(
                    f"[Gemini] Unexpected error with {model}: "
                    f"{type(exc).__name__}: {exc}"
                )

                # Move to the next model instead of killing
                # the entire ComicCraft pipeline.
                break

    # Every model failed.
    recent_failures = failures[-10:]

    details = "\n".join(
        f"  - {failure}"
        for failure in recent_failures
    )

    raise RuntimeError(
        "ComicCraft could not generate the requested Gemini content.\n\n"
        f"Models tried:\n"
        + "\n".join(f"  - {model}" for model in models)
        + "\n\nRecent errors:\n"
        + details
        + "\n\n"
        "This may be caused by temporary Gemini availability, "
        "quota limits, or model access restrictions."
    )