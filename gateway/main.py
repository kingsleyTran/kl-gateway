"""
OpenClaw Gateway — main entrypoint
Tự detect bootstrap state khi khởi động.
"""

import os
import json
from pathlib import Path
from enum import Enum

import chromadb
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env nếu có
ENV_FILE = Path("/config/.env")
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

# ─────────────────────────────────────────
# Bootstrap state check
# ─────────────────────────────────────────
def is_bootstrapped() -> bool:
    """Gateway coi là đã bootstrap nếu có ít nhất 1 provider key."""
    has_openai   = bool(os.getenv("OPENAI_API_KEY"))
    has_gemini   = bool(os.getenv("GEMINI_API_KEY"))
    has_copilot  = bool(os.getenv("COPILOT_TOKEN"))
    return any([has_openai, has_gemini, has_copilot])


# ─────────────────────────────────────────
# Config (chỉ load sau bootstrap)
# ─────────────────────────────────────────
CHROMA_HOST      = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT      = int(os.getenv("CHROMA_PORT", 8000))
OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "ollama")
OLLAMA_PORT      = int(os.getenv("OLLAMA_PORT", 11434))
EMBED_MODEL      = os.getenv("EMBED_MODEL", "nomic-embed-text")
PROJECTS_ROOT    = os.getenv("PROJECTS_ROOT", "/projects")
DEFAULT_MODEL    = os.getenv("DEFAULT_MODEL", "gpt-4.1")

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
COPILOT_TOKEN    = os.getenv("COPILOT_TOKEN")
COPILOT_API_URL  = os.getenv("COPILOT_API_URL", "http://copilot-api:4000")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

app = FastAPI(title="OpenClaw Gateway")

# Serve bootstrap UI static files
app.mount("/bootstrap", StaticFiles(directory="/app/bootstrap/static", html=True), name="bootstrap")

# ─────────────────────────────────────────
# Import routers
# ─────────────────────────────────────────
from routers import bootstrap_router, task_router

app.include_router(bootstrap_router.router, prefix="/api/bootstrap", tags=["bootstrap"])

@app.get("/")
async def root():
    if not is_bootstrapped():
        return FileResponse("/app/bootstrap/static/index.html")
    return {"status": "ready", "bootstrapped": True}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "bootstrapped": is_bootstrapped(),
    }

# Task routes chỉ available sau khi bootstrap
if is_bootstrapped():
    app.include_router(task_router.router, tags=["tasks"])
else:
    @app.get("/status")
    async def not_ready():
        return {"bootstrapped": False, "message": "Please complete setup at http://localhost:8080"}
