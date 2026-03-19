"""
Task router — RAG + model routing + diff validation.
Chỉ được mount sau khi bootstrap hoàn tất.
"""

import os
import json
from pathlib import Path
from enum import Enum

import chromadb
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path("/config/.env"))

router = APIRouter()

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
CHROMA_HOST    = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT    = int(os.getenv("CHROMA_PORT", 8000))
OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "ollama")
OLLAMA_PORT    = int(os.getenv("OLLAMA_PORT", 11434))
EMBED_MODEL    = os.getenv("EMBED_MODEL", "nomic-embed-text")
PROJECTS_ROOT  = os.getenv("PROJECTS_ROOT", "/projects")
DEFAULT_MODEL  = os.getenv("DEFAULT_MODEL", "gpt-4.1")

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
COPILOT_TOKEN   = os.getenv("COPILOT_TOKEN")
COPILOT_API_URL = os.getenv("COPILOT_API_URL", "http://copilot-api:4000")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Lazy clients
openai_client    = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
copilot_client   = OpenAI(api_key=COPILOT_TOKEN, base_url=f"{COPILOT_API_URL}/v1") if COPILOT_TOKEN else None

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


# ─────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────
class ModelProvider(str, Enum):
    OPENAI    = "openai"
    ANTHROPIC = "anthropic"
    GEMINI    = "gemini"
    COPILOT   = "copilot"

MODEL_REGISTRY: dict[str, ModelProvider] = {
    # OpenAI
    "gpt-4.1":           ModelProvider.OPENAI,
    "gpt-4.1-mini":      ModelProvider.OPENAI,
    "gpt-5.4":           ModelProvider.OPENAI,
    "codex-mini-latest": ModelProvider.OPENAI,
    # Anthropic
    "claude-sonnet-4-6": ModelProvider.ANTHROPIC,
    "claude-haiku-4-5":  ModelProvider.ANTHROPIC,
    # Gemini
    "gemini-2.5-pro":    ModelProvider.GEMINI,
    "gemini-2.5-flash":  ModelProvider.GEMINI,
    "gemini-3.1-pro":    ModelProvider.GEMINI,
    # Copilot (OpenAI-compatible via sidecar)
    "copilot-gpt-4.1":   ModelProvider.COPILOT,
    "copilot-claude":     ModelProvider.COPILOT,
}


# ─────────────────────────────────────────
# Project helpers
# ─────────────────────────────────────────
def list_projects() -> list[str]:
    root = Path(PROJECTS_ROOT)
    return [p.name for p in root.iterdir() if p.is_dir()] if root.exists() else []


def get_project_path(project: str) -> Path:
    path = Path(PROJECTS_ROOT) / project
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project}' not found. Available: {list_projects()}"
        )
    return path


# ─────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────
class TaskRequest(BaseModel):
    task: str
    project: str
    model: str = DEFAULT_MODEL
    top_k: int = 5

class ApplyConfirm(BaseModel):
    project: str
    file_path: str


# ─────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────
async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


# ─────────────────────────────────────────
# RAG
# ─────────────────────────────────────────
def get_or_create_collection(name: str):
    return chroma_client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


async def index_file(project: str, file_path: str):
    project_path = get_project_path(project)
    full_path = project_path / file_path
    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")

    content = full_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()
    chunks = ["\n".join(lines[i:i+100]) for i in range(0, len(lines), 100)]

    collection = get_or_create_collection(project)
    existing = collection.get(where={"file": file_path})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    for idx, chunk in enumerate(chunks):
        embedding = await get_embedding(chunk)
        collection.upsert(
            ids=[f"{file_path}::chunk_{idx}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"file": file_path, "chunk_idx": idx}],
        )
    return {"project": project, "indexed": file_path, "chunks": len(chunks)}


async def query_rag(task: str, project: str, top_k: int) -> list[dict]:
    embedding = await get_embedding(task)
    collection = get_or_create_collection(project)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"file": m["file"], "chunk_idx": m["chunk_idx"], "content": d, "score": round(1-dist, 4)}
        for d, m, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]


# ─────────────────────────────────────────
# Model Router
# ─────────────────────────────────────────
SYSTEM_PROMPT = """You are a code editing assistant.
Return ONLY a JSON object:
{
  "file": "<relative path>",
  "changes": [{"type": "replace", "start_line": <int>, "end_line": <int>, "new_content": "<code>"}]
}
No explanation. No markdown. Raw JSON only."""


def build_prompt(task: str, chunks: list[dict]) -> str:
    ctx = "\n\n".join(f"### {c['file']} chunk {c['chunk_idx']}\n{c['content']}" for c in chunks)
    return f"Task: {task}\n\nContext:\n{ctx}\n\nReturn JSON diff:"


def parse_json(raw: str, model: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        raise HTTPException(500, detail=f"[{model}] invalid JSON: {e}\nRaw: {raw}")


def call_model(task: str, chunks: list[dict], model: str) -> dict:
    if model not in MODEL_REGISTRY:
        raise HTTPException(400, detail=f"Unknown model '{model}'. Available: {list(MODEL_REGISTRY)}")

    provider = MODEL_REGISTRY[model]
    prompt = build_prompt(task, chunks)

    # OpenAI
    if provider == ModelProvider.OPENAI:
        if not openai_client:
            raise HTTPException(500, detail="OPENAI_API_KEY not configured. Re-run bootstrap.")
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        return parse_json(resp.choices[0].message.content, model)

    # Copilot (OpenAI-compatible sidecar)
    if provider == ModelProvider.COPILOT:
        if not copilot_client:
            raise HTTPException(500, detail="Copilot not configured. Re-run bootstrap.")
        resp = copilot_client.chat.completions.create(
            model=model.replace("copilot-", ""),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        return parse_json(resp.choices[0].message.content, model)

    # Anthropic
    if provider == ModelProvider.ANTHROPIC:
        if not ANTHROPIC_API_KEY:
            raise HTTPException(500, detail="ANTHROPIC_API_KEY not configured.")
        import anthropic as ac
        client = ac.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=model, max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return parse_json(resp.content[0].text, model)

    # Gemini
    if provider == ModelProvider.GEMINI:
        if not GEMINI_API_KEY:
            raise HTTPException(500, detail="GEMINI_API_KEY not configured.")
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        m = genai.GenerativeModel(model_name=model, system_instruction=SYSTEM_PROMPT)
        resp = m.generate_content(prompt)
        return parse_json(resp.text, model)

    raise HTTPException(500, detail=f"Unhandled provider: {provider}")


# ─────────────────────────────────────────
# Diff validation
# ─────────────────────────────────────────
def validate_diff(diff: dict, project: str) -> bool:
    path = get_project_path(project) / diff["file"]
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return all(c["end_line"] <= len(lines) for c in diff.get("changes", []))


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@router.post("/task")
async def handle_task(req: TaskRequest):
    chunks = await query_rag(req.task, req.project, req.top_k)
    if not chunks:
        raise HTTPException(404, detail=f"No code found in '{req.project}'. Run /index/full first.")
    diff = call_model(req.task, chunks, req.model)
    if not validate_diff(diff, req.project):
        raise HTTPException(409, detail="Diff validation failed. Re-index and retry.")
    return {"diff": diff, "model_used": req.model, "project": req.project, "rag_sources": [c["file"] for c in chunks]}


@router.post("/confirm-apply")
async def confirm_apply(req: ApplyConfirm):
    result = await index_file(req.project, req.file_path)
    return {"status": "re-indexed", **result}


@router.post("/index/file")
async def index_single_file(project: str, file_path: str):
    return await index_file(project, file_path)


@router.post("/index/full")
async def index_full(project: str, extensions: str = ".py,.ts,.js,.go,.java,.rs,.tsx,.jsx"):
    ext_list = extensions.split(",")
    project_path = get_project_path(project)
    indexed, errors = [], []
    for ext in ext_list:
        for f in project_path.rglob(f"*{ext}"):
            rel = str(f.relative_to(project_path))
            try:
                await index_file(project, rel)
                indexed.append(rel)
            except Exception as e:
                errors.append({"file": rel, "error": str(e)})
    return {"project": project, "indexed": len(indexed), "errors": errors}


@router.get("/projects")
async def get_projects():
    return {"projects": list_projects()}


@router.get("/models")
async def get_models():
    return {
        "models": list(MODEL_REGISTRY.keys()),
        "default": DEFAULT_MODEL,
        "by_provider": {
            p.value: [m for m, pv in MODEL_REGISTRY.items() if pv == p]
            for p in ModelProvider
        },
    }
