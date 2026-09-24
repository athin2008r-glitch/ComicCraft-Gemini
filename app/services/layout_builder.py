from __future__ import annotations

from typing import Any


def build_comic_layout(
    outline: list[dict[str, Any]],
    story: list[dict[str, Any]],
    image_results: list[str | dict[str, str]],
) -> list[dict[str, Any]]:
    """Merge outline, generated story text and generated asset paths into one render model."""
    if len(outline) != 5 or len(story) != 5 or len(image_results) != 5:
        raise ValueError("ComicCraft requires exactly five outline panels, story panels and images.")

    story_by_number = {int(item["panel_number"]): item for item in story}
    layout: list[dict[str, Any]] = []

    for index, panel in enumerate(outline, start=1):
        number = int(panel["panel_number"])
        story_panel = story_by_number.get(number)
        if not story_panel:
            raise ValueError(f"Missing story content for panel {number}.")
        image_result = image_results[index - 1]
        if isinstance(image_result, str):
            image_path = image_result
            image_provider = ""
        else:
            image_path = image_result["path"]
            image_provider = image_result.get("provider", "")
        layout.append(
            {
                "panel_number": number,
                "title": panel["title"],
                "scene_description": panel["scene_description"],
                "image_prompt": panel["image_prompt"],
                "image_path": image_path,
                "image_provider": image_provider,
                "narration": story_panel.get("narration", "").strip(),
                "dialogue": story_panel.get("dialogue", "").strip(),
            }
        )

    return layout
