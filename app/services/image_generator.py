from __future__ import annotations

import io
import os
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

BASE_DIR = Path(__file__).resolve().parents[2]
PANELS_DIR = BASE_DIR / "static" / "panels"
PANELS_DIR.mkdir(parents=True, exist_ok=True)


def _validated_image(value: object) -> Image.Image:
    """Convert a provider response into a non-empty RGB PIL image."""
    if isinstance(value, Image.Image):
        image = value
    elif isinstance(value, (bytes, bytearray)):
        image = Image.open(io.BytesIO(value))
    else:
        raise ValueError("Image provider returned an unsupported image response.")

    image.load()
    if image.width < 1 or image.height < 1:
        raise ValueError("Image provider returned an empty image.")
    return image.convert("RGB")


def _status_code(exc: Exception) -> int | None:
    for attribute in ("status_code", "status", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    return None


def _safe_error(exc: Exception) -> str:
    message = str(exc)
    for secret_name in ("HF_TOKEN",):
        secret = os.getenv(secret_name, "").strip()
        if secret:
            message = message.replace(secret, "[redacted]")
    code = _status_code(exc)
    return f"HTTP {code}: {message}" if code else f"{type(exc).__name__}: {message}"


def _hf_image(prompt: str) -> Image.Image:
    """Generate through Hugging Face Inference Providers."""
    from huggingface_hub import InferenceClient

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not configured.")

    model = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
    client = InferenceClient(api_key=token, provider="auto")
    result = client.text_to_image(prompt=prompt, model=model)
    return _validated_image(result)


def _fallback_card(prompt: str, panel_number: int, art_style: str) -> Image.Image:
    """Last-resort local asset so the web/PDF pipeline stays testable when AI image APIs fail."""
    image = Image.new("RGB", (1024, 1024), "#f4efe4")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        title_font = None
        body_font = None
    draw.rounded_rectangle((36, 36, 988, 988), radius=28, outline="#222", width=8)
    draw.text((80, 90), f"ComicCraft • Panel {panel_number}", fill="#111", font=title_font)
    draw.text((80, 165), f"Style: {art_style}", fill="#333", font=body_font)
    snippet = prompt.replace("\n", " ")[:420]
    lines = []
    words = snippet.split()
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > 48:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    y = 250
    for line in lines[:8]:
        draw.text((80, y), line, fill="#222", font=body_font)
        y += 42
    draw.text((80, 840), "Image provider unavailable — pipeline fallback asset", fill="#666", font=body_font)
    return image


def generate_image(
    prompt: str,
    panel_number: int,
    art_style: str,
    character_context: str = "",
) -> dict[str, str]:
    """Generate a panel and return its relative path and provider name."""
    enhanced_prompt = (
        f"{prompt}. Main character continuity: {character_context or 'preserve the character details in the prompt'}. "
        f"Art style: {art_style}. Clean comic-book aesthetic, cinematic composition, appropriate environment, "
        "dramatic lighting where appropriate, high visual clarity, consistent character appearance, "
        "no text, no speech bubbles, no captions, no watermark, no logo."
    )

    backend = os.getenv("IMAGE_BACKEND", "auto").lower().strip()
    if backend not in {"auto", "hf", "openai", "local"}:
        print(f"[Image] Unknown IMAGE_BACKEND={backend!r}; using auto.")
        backend = "auto"

    image: Image.Image | None = None
    provider = "local fallback"

    providers = []
    if backend in {"auto", "hf"}:
        providers.append(("Hugging Face", _hf_image))
    for name, generator in providers:
        try:
            image = generator(enhanced_prompt)
            image = _validated_image(image)
            provider = name
            print(f"[Image] Panel {panel_number}: {name} succeeded")
            break
        except Exception as exc:
            message = _safe_error(exc)
            if name == "Hugging Face" and _status_code(exc) == 403:
                message = "HTTP 403: token does not have sufficient Inference Provider permissions"
            print(f"[Image] Panel {panel_number}: {name} failed ({message})")

    if image is None:
        print(f"[Image] Panel {panel_number}: Using local fallback")
        image = _fallback_card(enhanced_prompt, panel_number, art_style)

    image = _validated_image(image)
    filename = f"panel_{int(time.time() * 1000)}_{panel_number}_{uuid4().hex[:8]}.png"
    destination = PANELS_DIR / filename
    image.save(destination, format="PNG", optimize=True)

    return {
        "path": str(Path("static") / "panels" / filename).replace("\\", "/"),
        "provider": provider,
    }
