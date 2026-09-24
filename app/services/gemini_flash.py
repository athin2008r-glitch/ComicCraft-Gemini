from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .ai_text_runtime import TextGenerationResult, generate_structured_with_fallback


class PanelOutline(BaseModel):
    panel_number: int = Field(ge=1, le=5)
    title: str = Field(min_length=1, max_length=100)
    scene_description: str = Field(min_length=1, max_length=900)
    image_prompt: str = Field(min_length=1, max_length=1200)


class OutlineResponse(BaseModel):
    panels: list[PanelOutline]


OUTLINE_FALLBACKS = (
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
)
def _local_outline(
    prompt: str,
    character_name: str,
    setting: str,
    tone: str,
    art_style: str,
) -> OutlineResponse:
    phases = (
        ("A New Beginning", f"{character_name} starts a {tone.lower()} adventure in the {setting}.", "wide establishing shot"),
        ("A Strange Discovery", f"{character_name} discovers an unexpected clue while exploring the {setting}.", "medium discovery shot"),
        ("The Challenge", f"A sudden obstacle tests {character_name} and raises the stakes.", "dynamic action shot"),
        ("The Turning Point", f"{character_name} makes a brave choice and faces the challenge directly.", "dramatic close-up"),
        ("A Hopeful Ending", f"{character_name} resolves the adventure and finds a hopeful path forward.", "warm final wide shot"),
    )
    return OutlineResponse(panels=[
        PanelOutline(
            panel_number=index,
            title=title,
            scene_description=description,
            image_prompt=f"{shot} of {character_name} in the {setting}, {description} Art style: {art_style}.",
        )
        for index, (title, description, shot) in enumerate(phases, start=1)
    ])


def generate_outline_with_provider(
    prompt: str,
    character_name: str,
    setting: str,
    tone: str,
    art_style: str,
) -> tuple[list[dict[str, Any]], str]:
    """Generate the outline and report which text provider was used."""
    user_prompt = f"""
Create a coherent 5-panel comic storyboard.

Story premise: {prompt}
Main character: {character_name}
Setting: {setting}
Tone: {tone}
Art style: {art_style}

Requirements:
- Exactly 5 panels, numbered 1 through 5.
- The story must have a clear beginning, development, turning point, and ending.
- Keep the main character visually and narratively consistent.
- image_prompt describes only the illustration, with no written text, captions, speech bubbles, logos, or watermarks.
- Return only the requested structured data.
""".strip()
    result: TextGenerationResult = generate_structured_with_fallback(
        prompt=user_prompt,
        response_model=OutlineResponse,
        gemini_models=OUTLINE_FALLBACKS,
        local_fallback=lambda: _local_outline(prompt, character_name, setting, tone, art_style),
        gemini_env_name="GEMINI_OUTLINE_MODEL",
    )
    panels = sorted(result.value.panels, key=lambda panel: panel.panel_number)
    if len(panels) != 5 or [panel.panel_number for panel in panels] != [1, 2, 3, 4, 5]:
        raise RuntimeError("Text provider did not return exactly five ordered panels.")
    return [panel.model_dump() for panel in panels], result.provider


def generate_outline(
    prompt: str,
    character_name: str,
    setting: str,
    tone: str,
    art_style: str,
) -> list[dict[str, Any]]:
    return generate_outline_with_provider(prompt, character_name, setting, tone, art_style)[0]
