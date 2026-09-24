from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from .schemas import PromptRequest
from .services.exporters import save_pdf
from .services.gemini_flash import generate_outline_with_provider
from .services.gemini_pro import generate_story_with_provider
from .services.image_generator import generate_image
from .services.layout_builder import build_comic_layout

BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
router = APIRouter()


def _project_relative_static_path(value: str) -> Path:
    """Resolve a static path safely and reject traversal outside static/."""
    raw = Path(value)
    if raw.is_absolute() or raw.parts[:1] != ("static",):
        raise HTTPException(status_code=400, detail="Invalid static file path.")
    candidate = (BASE_DIR / raw).resolve()
    static_root = (BASE_DIR / "static").resolve()
    try:
        candidate.relative_to(static_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid static file path.") from exc
    return candidate


def _run_pipeline(request: PromptRequest) -> dict[str, Any]:
    outline, outline_provider = generate_outline_with_provider(
        request.prompt,
        request.character_name,
        request.setting,
        request.tone,
        request.art_style,
    )
    story, story_provider = generate_story_with_provider(outline, request.character_name, request.tone)
    image_results = [
        generate_image(
            panel["image_prompt"],
            panel["panel_number"],
            request.art_style,
            character_context=f"{request.character_name} in the {request.setting} setting",
        )
        for panel in outline
    ]
    layout = build_comic_layout(outline, story, image_results)
    title = f"{request.character_name}'s Comic Adventure"
    pdf_path = save_pdf(layout, title)
    return {
        "title": title,
        "request": request.model_dump(),
        "layout": layout,
        "pdf_path": pdf_path,
        "pdf_filename": Path(pdf_path).name,
        "text_provider": outline_provider if outline_provider == story_provider else f"{outline_provider} + {story_provider}",
    }


@router.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@router.post("/generate")
def generate_comic(
    request: Request,
    prompt: str = Form(...),
    character_name: str = Form(...),
    setting: str = Form(...),
    tone: str = Form(...),
    art_style: str = Form(...),
):
    try:
        validated = PromptRequest(
            prompt=prompt,
            character_name=character_name,
            setting=setting,
            tone=tone,
            art_style=art_style,
        )
        result = _run_pipeline(validated)
    except ValidationError as exc:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": "Please correct the form values.",
                "validation": exc.errors(),
            },
            status_code=422,
        )
    except Exception as exc:
        print(f"[ComicCraft] Generation failed: {type(exc).__name__}")
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"error": "Comic generation could not be completed. Please try again."},
            status_code=500,
        )

    return templates.TemplateResponse(
        request=request,
        name="comic_preview.html",
        context={
            "title": result["title"],
            "layout": result["layout"],
            "pdf_path": result["pdf_path"],
            "text_provider": result["text_provider"],
        },
    )


@router.post("/generate-comic/json")
def generate_comic_json(payload: PromptRequest):
    try:
        result = _run_pipeline(payload)
        return JSONResponse(
            content={
                "success": True,
                "title": result["title"],
                "panels": result["layout"],
                "provider": result["text_provider"],
                "pdf_url": f"/download-pdf?file={result['pdf_filename']}",
            }
        )
    except Exception as exc:
        print(f"[ComicCraft] JSON generation failed: {type(exc).__name__}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Comic generation could not be completed."},
        )


@router.get("/download-pdf")
def download_pdf(file: str):
    candidate = _project_relative_static_path(f"static/exports/{Path(file).name}")
    if candidate.parent != (BASE_DIR / "static" / "exports").resolve() or not candidate.exists():
        raise HTTPException(status_code=404, detail="PDF not found.")
    return FileResponse(
        path=str(candidate),
        media_type="application/pdf",
        filename=candidate.name,
        headers={"Content-Disposition": f'attachment; filename="{candidate.name}"'},
    )


@router.get("/export-success")
def export_success(request: Request, file: str | None = None):
    pdf_filename = Path(file).name if file else None
    return templates.TemplateResponse(
        request=request,
        name="export_success.html",
        context={
            "pdf_filename": pdf_filename,
            "pdf_url": f"/download-pdf?file={pdf_filename}" if pdf_filename else None,
        },
    )


@router.post("/test-image")
def test_image(prompt: str = Form(...), art_style: str = Form("comic book")):
    if not prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt is required.")
    try:
        result = generate_image(prompt.strip(), 0, art_style.strip() or "comic book")
    except Exception as exc:
        print(f"[Image] Test image failed: {type(exc).__name__}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Image generation failed."},
        )
    return JSONResponse(
        {
            "success": True,
            "path": result["path"],
            "provider": result["provider"],
            "url": "/" + result["path"],
        }
    )
