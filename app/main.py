from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routes import router

BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "panels").mkdir(exist_ok=True)
(STATIC_DIR / "exports").mkdir(exist_ok=True)

app = FastAPI(
    title="ComicCraft - AI Comic Story Creator",
    version="1.0.0",
    description="Generate a five-panel comic using Gemini text generation and AI image inference.",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(router)
