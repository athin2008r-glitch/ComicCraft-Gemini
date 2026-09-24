# ComicCraft - AI Comic Story Creator using Gemini Models

ComicCraft is a five-panel comic generator built for a college capstone. A user supplies a story prompt, character name, setting, tone, and art style. The backend then:

1. Creates a five-panel outline with Gemini Flash, OpenRouter free text fallback, or local fallback.
2. Creates narration and dialogue with the same Gemini -> OpenRouter -> local fallback chain.
3. Generates one illustration per panel.
4. Builds a structured panel layout for Jinja2.
5. Compiles the result to an A4 PDF with fpdf2.

## Important 2026 compatibility note

The original faculty specification names `gemini-1.5-flash`, `gemini-1.5-pro`, and the legacy `google-generativeai` package. Those are historical dependencies. The implementation uses Google's current `google-genai` SDK and configurable current model IDs instead. This preserves the requested Flash/Pro division while avoiding a dependency on retired model endpoints.

Defaults:

- Outline and story model names are configurable through `GEMINI_OUTLINE_MODEL` and `GEMINI_STORY_MODEL`.
- Text uses Gemini first, then OpenRouter's free model router, then a deterministic local five-panel fallback.
- Images use `IMAGE_BACKEND=hf` with Hugging Face, or the local fallback card when `auto` is selected.
- Hugging Face defaults to `black-forest-labs/FLUX.1-schnell`.

The model IDs are configured through environment variables so a faculty-approved model can be swapped without editing application code.

## Directory structure

```text
comiccraft/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── routes.py
│   ├── schemas.py
│   └── services/
│       ├── __init__.py
│       ├── gemini_flash.py
│       ├── gemini_pro.py
│       ├── ai_text_runtime.py
│       ├── image_generator.py
│       ├── layout_builder.py
│       └── exporters.py
├── templates/
│   ├── index.html
│   ├── comic_preview.html
│   └── export_success.html
├── static/
│   ├── css/styles.css
│   ├── panels/
│   └── exports/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Windows setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and add the keys you have:

```text
GEMINI_API_KEY=...
HF_TOKEN=...
OPENROUTER_API_KEY=...
OPENROUTER_TEXT_MODEL=openrouter/free
IMAGE_BACKEND=hf
```

API keys must never be committed to GitHub. Text generation automatically tries Gemini, then OpenRouter when `OPENROUTER_API_KEY` is present, then the deterministic local fallback. Image generation uses Hugging Face, then the local fallback card when `IMAGE_BACKEND=auto`. You can select an image backend explicitly with `hf` or `local`.

Start the server:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` and API docs at `http://127.0.0.1:8000/docs`.

## Dependencies

Install all required packages with:

```powershell
pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, Jinja2, python-multipart, Pydantic, the Google GenAI SDK, Hugging Face Hub, Pillow, fpdf2, python-dotenv, and the OpenAI-compatible SDK used to connect to OpenRouter. Local PyTorch/Diffusers packages are deliberately not required for the default remote-provider workflow.

## Android/mobile development

Yes. The application is very suitable for phone-based development because FastAPI, Jinja2, Pillow and fpdf2 are ordinary Python web components. Termux can provide a Python development environment on Android.

Recommended Android architecture:

```text
Android phone
   |
   +-- Termux / Git / Python / Uvicorn
   |
   +-- ComicCraft FastAPI server
           |
           +-- Gemini API (outline + story), then OpenAI text fallback
           |
           +-- Hugging Face Inference (panel images)
           |
           +-- Pillow + fpdf2 (local layout + PDF)
```

### Android setup

Install Termux from a trusted/current source, then:

```bash
pkg update && pkg upgrade -y
pkg install python git -y
python --version
python -m pip install --upgrade pip
```

Create/open the project and install dependencies:

```bash
cd ~/ComicCraft-Gemini
python -m venv comiccraft-env
source comiccraft-env/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Run:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` in the phone browser.

### GitHub from the phone

Git works from Termux, so the complete source tree can be committed and pushed from Android:

```bash
git init
git add .
git commit -m "Initial ComicCraft capstone"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY>
git push -u origin main
```

## Environment variables

See `.env.example`.

Do not commit `.env` or generated `.png`/`.pdf` assets. The included `.gitignore` prevents that.

## API routes

- `GET /` - form page
- `POST /generate` - end-to-end web pipeline
- `POST /generate-comic/json` - JSON API
- `GET /download-pdf?file=...` - PDF download
- `GET /export-success?file=...` - export confirmation
- `POST /test-image` - standalone image-generation test
- `GET /docs` - FastAPI Swagger documentation

## Verification

Use this faculty-style test input:

- Story: `A brave fox exploring an enchanted forest`
- Character: `Rox`
- Tone: `Dramatic`
- Style: `Anime`

A successful run should produce five panel cards, five panel images, and one PDF under `static/exports/`.

## Error handling behavior

If the Gemini configuration is missing or an API call fails, the UI returns the error instead of silently claiming a successful AI generation.

Image failures are logged server-side with the provider name and a safe reason. API keys are never returned to the browser. In automatic mode, a Hugging Face permission/quota/provider/network failure moves to OpenAI; if that also fails, the clearly labeled local fallback asset keeps the comic and PDF pipeline usable.
