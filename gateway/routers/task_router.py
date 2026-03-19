"""
Task router — RAG + model routing.
API keys đọc từ SQLite mỗi request, không cache vào biến module.
"""

import os
import json
import time
from pathlib import Path
from enum import Enum

import chromadb
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db import (
    get_config, get_project, update_project_indexed,
    list_projects as db_list_projects
)

router = APIRouter()

# Infra config từ env (stable, không thay đổi sau deploy)
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
HOST_PREFIX = os.getenv("HOST_MOUNT_PREFIX", "")

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


# ── Model Registry ────────────────────────────────
class ModelProvider(str, Enum):
    OPENAI    = "openai"
    ANTHROPIC = "anthropic"
    GEMINI    = "gemini"
    COPILOT   = "copilot"

MODEL_REGISTRY: dict[str, ModelProvider] = {
    "gpt-4.1":           ModelProvider.OPENAI,
    "gpt-4.1-mini":      ModelProvider.OPENAI,
    "gpt-5.4":           ModelProvider.OPENAI,
    "codex-mini-latest": ModelProvider.OPENAI,
    "claude-sonnet-4-6": ModelProvider.ANTHROPIC,
    "claude-haiku-4-5":  ModelProvider.ANTHROPIC,
    "gemini-2.5-pro":    ModelProvider.GEMINI,
    "gemini-2.5-flash":  ModelProvider.GEMINI,
    "gemini-3.1-pro":    ModelProvider.GEMINI,
    "copilot-gpt-4.1":  ModelProvider.COPILOT,
    "copilot-claude":    ModelProvider.COPILOT,
}


# ── Schemas ───────────────────────────────────────
class TaskRequest(BaseModel):
    task:    str
    project: str
    model:   str = ""   # empty = use default_model from DB
    top_k:   int = 5

class ApplyConfirm(BaseModel):
    project:   str
    file_path: str


# ── Helpers ───────────────────────────────────────
def host_path(p: str) -> Path:
    return Path(HOST_PREFIX + str(Path(p).expanduser().resolve()))


async def resolve_project_path(name: str) -> Path:
    project = await get_project(name)
    if not project:
        projects = await db_list_projects()
        raise HTTPException(404, detail=f"Project '{name}' not found. Available: {[p['name'] for p in projects]}")
    path = host_path(project["path"])
    if not path.exists():
        raise HTTPException(400, detail=f"Project path no longer exists: {project['path']}")
    return path


# ── Embedding ─────────────────────────────────────
async def get_embedding(text: str) -> list[float]:
    embed_model = await get_config("embed_model", "nomic-embed-text")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
            json={"model": embed_model, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


# ── RAG ───────────────────────────────────────────
def get_or_create_collection(name: str):
    return chroma_client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


async def index_file(project_name: str, file_path: str, project_path: Path) -> int:
    full_path = project_path / file_path
    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")

    content = full_path.read_text(encoding="utf-8", errors="ignore")
    lines   = content.splitlines()
    chunks  = ["\n".join(lines[i:i+100]) for i in range(0, len(lines), 100)]

    collection = get_or_create_collection(project_name)
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
    return len(chunks)


async def query_rag(task: str, project_name: str, top_k: int) -> list[dict]:
    embedding  = await get_embedding(task)
    collection = get_or_create_collection(project_name)
    results    = collection.query(
        query_embeddings=[embedding], n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"file": m["file"], "chunk_idx": m["chunk_idx"], "content": d, "score": round(1-dist, 4)}
        for d, m, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


# ── Model Router ──────────────────────────────────
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
        if raw.startswith("json"): raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        raise HTTPException(500, detail=f"[{model}] invalid JSON: {e}\nRaw: {raw}")


async def call_model(task: str, chunks: list[dict], model: str) -> tuple[dict, dict]:
    """Baca API key từ SQLite mỗi lần call — không cache."""
    if model not in MODEL_REGISTRY:
        raise HTTPException(400, detail=f"Unknown model '{model}'. Available: {list(MODEL_REGISTRY)}")

    provider = MODEL_REGISTRY[model]
    prompt   = build_prompt(task, chunks)
    metrics  = {"tokens": {"prompt": 0, "completion": 0, "total": 0}, "provider": provider.value}

    # ── OpenAI ──────────────────────────────────
    if provider == ModelProvider.OPENAI:
        api_key = await get_config("openai_key")
        if not api_key:
            raise HTTPException(500, detail="OpenAI not configured. Re-run bootstrap.")
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        if resp.usage:
            metrics["tokens"] = {"prompt": resp.usage.prompt_tokens,
                                 "completion": resp.usage.completion_tokens,
                                 "total": resp.usage.total_tokens}
        return parse_json(resp.choices[0].message.content, model), metrics

    # ── Copilot ──────────────────────────────────
    if provider == ModelProvider.COPILOT:
        token = await get_config("copilot_token")
        copilot_url = os.getenv("COPILOT_API_URL", "http://localhost:4000")
        if not token:
            raise HTTPException(500, detail="Copilot not configured. Re-run bootstrap.")
        from openai import OpenAI
        client = OpenAI(api_key=token, base_url=f"{copilot_url}/v1")
        resp = client.chat.completions.create(
            model=model.replace("copilot-", ""),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        if resp.usage:
            metrics["tokens"] = {"prompt": resp.usage.prompt_tokens,
                                 "completion": resp.usage.completion_tokens,
                                 "total": resp.usage.total_tokens}
        return parse_json(resp.choices[0].message.content, model), metrics

    # ── Anthropic ─────────────────────────────────
    if provider == ModelProvider.ANTHROPIC:
        api_key = await get_config("anthropic_key")
        if not api_key:
            raise HTTPException(500, detail="Anthropic not configured. Re-run bootstrap.")
        import anthropic as ac
        client = ac.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model, max_tokens=2000, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        metrics["tokens"] = {"prompt": resp.usage.input_tokens,
                             "completion": resp.usage.output_tokens,
                             "total": resp.usage.input_tokens + resp.usage.output_tokens}
        return parse_json(resp.content[0].text, model), metrics

    # ── Gemini ────────────────────────────────────
    if provider == ModelProvider.GEMINI:
        api_key = await get_config("gemini_key")
        if not api_key:
            raise HTTPException(500, detail="Gemini not configured. Re-run bootstrap.")
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model_name=model, system_instruction=SYSTEM_PROMPT)
        resp = m.generate_content(prompt)
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            metrics["tokens"] = {"prompt": resp.usage_metadata.prompt_token_count,
                                 "completion": resp.usage_metadata.candidates_token_count,
                                 "total": resp.usage_metadata.total_token_count}
        return parse_json(resp.text, model), metrics

    raise HTTPException(500, detail=f"Unhandled provider: {provider}")


# ── Diff validation ───────────────────────────────
async def validate_diff(diff: dict, project_path: Path) -> bool:
    path = project_path / diff["file"]
    if not path.exists(): return False
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return all(c["end_line"] <= len(lines) for c in diff.get("changes", []))


# ── Routes ────────────────────────────────────────
@router.post("/task")
async def handle_task(req: TaskRequest):
    project_path = await resolve_project_path(req.project)

    # Resolve model — fallback về default từ DB
    model = req.model or await get_config("default_model", "gpt-4.1")

    timings = {}
    t0 = time.perf_counter()

    t_rag = time.perf_counter()
    chunks = await query_rag(req.task, req.project, req.top_k)
    timings["rag_ms"] = round((time.perf_counter() - t_rag) * 1000)

    if not chunks:
        raise HTTPException(404, detail=f"No code found in '{req.project}'. Run /index/full?project={req.project} first.")

    t_model = time.perf_counter()
    diff, token_metrics = await call_model(req.task, chunks, model)
    timings["model_ms"] = round((time.perf_counter() - t_model) * 1000)

    if not await validate_diff(diff, project_path):
        raise HTTPException(409, detail="Diff validation failed — file may have changed. Re-index and retry.")

    timings["total_ms"] = round((time.perf_counter() - t0) * 1000)

    return {
        "diff":    diff,
        "project": req.project,
        "metrics": {
            "model":       model,
            "provider":    token_metrics["provider"],
            "tokens":      token_metrics["tokens"],
            "timings":     timings,
            "rag_sources": [{"file": c["file"], "score": c["score"]} for c in chunks],
            "rag_chunks":  len(chunks),
        },
    }


@router.post("/confirm-apply")
async def confirm_apply(req: ApplyConfirm):
    project_path = await resolve_project_path(req.project)
    chunks = await index_file(req.project, req.file_path, project_path)
    return {"status": "re-indexed", "project": req.project, "file": req.file_path, "chunks": chunks}


@router.post("/index/file")
async def index_single_file(project: str, file_path: str):
    project_path = await resolve_project_path(project)
    chunks = await index_file(project, file_path, project_path)
    return {"project": project, "indexed": file_path, "chunks": chunks}


@router.post("/index/full")
async def index_full(project: str, extensions: str = ".py,.ts,.js,.go,.java,.rs,.tsx,.jsx"):
    project_path = await resolve_project_path(project)
    indexed, errors = [], []
    for ext in extensions.split(","):
        for f in project_path.rglob(f"*{ext}"):
            rel = str(f.relative_to(project_path))
            try:
                await index_file(project, rel, project_path)
                indexed.append(rel)
            except Exception as e:
                errors.append({"file": rel, "error": str(e)})
    await update_project_indexed(project, len(indexed))
    return {"project": project, "indexed": len(indexed), "errors": errors}


@router.get("/projects")
async def get_projects():
    projects = await db_list_projects()
    return {"projects": [p["name"] for p in projects]}


@router.get("/models")
async def get_models():
    default = await get_config("default_model", "gpt-4.1")
    return {
        "models":  list(MODEL_REGISTRY.keys()),
        "default": default,
        "by_provider": {
            p.value: [m for m, pv in MODEL_REGISTRY.items() if pv == p]
            for p in ModelProvider
        },
    }
