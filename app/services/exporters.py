from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from uuid import uuid4

from fpdf import FPDF

BASE_DIR = Path(__file__).resolve().parents[2]
EXPORTS_DIR = BASE_DIR / "static" / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _pdf_text(value: str) -> str:
    """Convert arbitrary Unicode text into safe single-byte text for built-in FPDF fonts."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("latin-1", errors="replace").decode("latin-1")
    return ascii_text.replace("\x00", "").strip()


def _safe_title(title: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", title.strip())
    return cleaned.strip("_")[:60] or "ComicCraft_Comic"


def save_pdf(layout_data: list[dict], title: str) -> str:
    """Build an A4 PDF with one clean panel per page and return its static relative path."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_title(_pdf_text(title))
    pdf.set_creator("ComicCraft")

    for panel in layout_data:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 12, _pdf_text(f"Panel {panel['panel_number']}: {panel['title']}"), ln=True)

        image_path = BASE_DIR / panel["image_path"]
        if image_path.exists():
            x, y, w = 15, 30, 180
            max_h = 105
            try:
                pdf.image(str(image_path), x=x, y=y, w=w, h=max_h)
                cursor_y = y + max_h + 8
            except Exception:
                cursor_y = y
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(0, 8, "Image could not be embedded.", ln=True)
        else:
            cursor_y = 30
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 8, "Image file not found.", ln=True)

        pdf.set_y(cursor_y)

        description = _pdf_text(panel.get("scene_description", ""))
        narration = _pdf_text(panel.get("narration", ""))
        dialogue = _pdf_text(panel.get("dialogue", ""))

        content_width = 180

        if description:
            pdf.set_font("Helvetica", "I", 11)
            pdf.set_x(15)
            pdf.multi_cell(content_width, 7, description)
            pdf.ln(2)

        if narration:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_x(15)
            pdf.multi_cell(content_width, 7, "Narration")
            pdf.set_font("Helvetica", "", 11)
            pdf.set_x(15)
            pdf.multi_cell(content_width, 7, narration)
            pdf.ln(2)

        if dialogue:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_x(15)
            pdf.multi_cell(content_width, 7, "Dialogue")
            pdf.set_font("Helvetica", "", 11)
            pdf.set_x(15)
            pdf.multi_cell(content_width, 7, dialogue)

    filename = f"{_safe_title(title)}_{uuid4().hex[:10]}.pdf"
    destination = EXPORTS_DIR / filename
    pdf.output(str(destination))
    return str(Path("static") / "exports" / filename).replace("\\", "/")
