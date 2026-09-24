from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from .ai_text_runtime import TextGenerationResult, generate_structured_with_fallback


class StoryPanel(BaseModel):
    panel_number: int = Field(ge=1, le=5)
    narration: str = Field(default="", max_length=500)
    dialogue: str = Field(default="", max_length=700)


class StoryResponse(BaseModel):
    panels: list[StoryPanel]


STORY_FALLBACKS = (
    "gemini-3.1-pro-preview",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
)
def _local_story(outline: list[dict[str, Any]], character_name: str, tone: str) -> StoryResponse:
    narration = (
        f"{character_name}'s {tone.lower()} adventure begins.",
        f"A surprising clue pulls {character_name} deeper into the mystery.",
        f"The challenge grows, but {character_name} refuses to give up.",
        f"At the turning point, {character_name} finds the courage to act.",
        f"The adventure ends with {character_name} wiser and hopeful.",
    )
    dialogue = (
        "I wonder what comes next.",
        "This clue must mean something.",
        "I can handle this.",
        "Now is the moment to be brave.",
        "We made it through together!",
    )
    return StoryResponse(panels=[
        StoryPanel(panel_number=index, narration=narration[index - 1], dialogue=dialogue[index - 1])
        for index in range(1, 6)
    ])


def generate_story_with_provider(
    outline: list[dict[str, Any]],
    character_name: str,
    tone: str,
) -> tuple[list[dict[str, Any]], str]:
    serialized_outline = json.dumps(outline, ensure_ascii=False, indent=2)
    prompt = f"""
Write narration and dialogue for this 5-panel comic.

Main character: {character_name}
Tone: {tone}
Storyboard:
{serialized_outline}

Return exactly one object for each panel numbered 1 through 5. Keep narration concise, dialogue short, and continuity intact. Return only structured data.
""".strip()
    result: TextGenerationResult = generate_structured_with_fallback(
        prompt=prompt,
        response_model=StoryResponse,
        gemini_models=STORY_FALLBACKS,
        local_fallback=lambda: _local_story(outline, character_name, tone),
        gemini_env_name="GEMINI_STORY_MODEL",
    )
    panels = sorted(result.value.panels, key=lambda panel: panel.panel_number)
    if len(panels) != 5 or [panel.panel_number for panel in panels] != [1, 2, 3, 4, 5]:
        raise RuntimeError("Text provider did not return exactly five story panels.")
    return [panel.model_dump() for panel in panels], result.provider


def generate_story(
    outline: list[dict[str, Any]],
    character_name: str,
    tone: str,
) -> list[dict[str, Any]]:
    return generate_story_with_provider(outline, character_name, tone)[0]
