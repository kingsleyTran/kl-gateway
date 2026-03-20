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
from openai import APIError, AuthenticationError, NotFoundError, OpenAI, PermissionDeniedError, RateLimitError

from db import (
    get_config, get_project, set_config, update_project_indexed,
    list_projects as db_list_projects
)
from oauth import OPENAI_CODEX_API_BASE, refresh_openai_codex_token

router = APIRouter()

# Infra config từ env (stable, không thay đổi sau deploy)
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
HOST_PREFIX = os.getenv("HOST_MOUNT_PREFIX", "")

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# DEFAULT_MODEL không hardcode — luôn đọc từ DB (set lúc bootstrap)
# Dùng get_config("default_model") mỗi request thay vì cache ở đây


# ── Model Registry ────────────────────────────────
class ModelProvider(str, Enum):
    OPENAI = "openai"
    OPENAI_CODEX = "openai-codex"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    COPILOT = "copilot"

# Static registry — fallback và well-known models
MODEL_REGISTRY: dict[str, ModelProvider] = {
    # OpenAI API key
    "openai/gpt-4.1": ModelProvider.OPENAI,
    "openai/gpt-4.1-mini": ModelProvider.OPENAI,
    "openai/gpt-4o": ModelProvider.OPENAI,
    "openai/gpt-4o-mini": ModelProvider.OPENAI,
    "openai/gpt-5.4": ModelProvider.OPENAI,
    "openai/codex-mini-latest": ModelProvider.OPENAI,
    # OpenAI Codex OAuth
    "openai-codex/gpt-5.4": ModelProvider.OPENAI_CODEX,
    "openai-codex/gpt-5.3-codex": ModelProvider.OPENAI_CODEX,
    "openai-codex/gpt-5.3-codex-spark": ModelProvider.OPENAI_CODEX,
    # Anthropic
    "claude-sonnet-4-6":    ModelProvider.ANTHROPIC,
    "claude-haiku-4-5":     ModelProvider.ANTHROPIC,
    "claude-opus-4-6":      ModelProvider.ANTHROPIC,
    # Gemini — thêm nhiều versions
    "gemini-2.5-pro":       ModelProvider.GEMINI,
    "gemini-2.5-flash":     ModelProvider.GEMINI,
    "gemini-2.0-flash":     ModelProvider.GEMINI,
    "gemini-2.0-flash-001": ModelProvider.GEMINI,
    "gemini-2.0-flash-lite":ModelProvider.GEMINI,
    "gemini-1.5-pro":       ModelProvider.GEMINI,
    "gemini-1.5-flash":     ModelProvider.GEMINI,
    "gemini-3.1-pro":       ModelProvider.GEMINI,
    # Copilot
    "copilot-gpt-4.1":      ModelProvider.COPILOT,
    "copilot-claude":        ModelProvider.COPILOT,
}


def normalize_model_name(model: str) -> tuple[ModelProvider, str]:
    raw = (model or "").strip()
    if not raw:
        raise HTTPException(400, detail="Model is required.")
    if raw in MODEL_REGISTRY:
        provider = MODEL_REGISTRY[raw]
        return provider, raw.split("/", 1)[1] if "/" in raw else raw

    lower = raw.lower()
    if lower.startswith("openai-codex/"):
        return ModelProvider.OPENAI_CODEX, raw.split("/", 1)[1]
    if lower.startswith("openai/"):
        return ModelProvider.OPENAI, raw.split("/", 1)[1]
    if lower.startswith("anthropic/"):
        return ModelProvider.ANTHROPIC, raw.split("/", 1)[1]
    if lower.startswith("gemini/"):
        return ModelProvider.GEMINI, raw.split("/", 1)[1]
    if lower.startswith("copilot/"):
        return ModelProvider.COPILOT, raw.split("/", 1)[1]

    if lower.startswith("gpt-") or lower.startswith(("o1", "o3", "o4")):
        return ModelProvider.OPENAI, raw
    if lower.startswith("claude-"):
        return ModelProvider.ANTHROPIC, raw
    if lower.startswith("gemini-"):
        return ModelProvider.GEMINI, raw
    if lower.startswith("copilot-"):
        return ModelProvider.COPILOT, raw.replace("copilot-", "", 1)

    raise HTTPException(400, detail=f"Unknown model '{model}'. Use provider/model, for example openai/gpt-4.1 or openai-codex/gpt-5.4.")


def resolve_provider(model: str) -> ModelProvider:
    return normalize_model_name(model)[0]


# ── Schemas ───────────────────────────────────────
class TaskRequest(BaseModel):
    task:    str
    project: str
    model:   str = ""   # empty → đọc default_model từ DB
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
        try:
            resp = await client.post(
                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
                json={"model": embed_model, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except httpx.ConnectError:
            raise HTTPException(503, detail="Ollama is not running. Start it with: docker compose up ollama")
        except httpx.TimeoutException:
            raise HTTPException(504, detail="Ollama timed out. It may still be loading the model.")
        except Exception as e:
            raise HTTPException(503, detail=f"Ollama error: {str(e)}")


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

    # n_results không được vượt quá số chunks thực tế trong collection
    count = collection.count()
    if count == 0:
        return []
    n = min(top_k, count)

    results    = collection.query(
        query_embeddings=[embedding], n_results=n,
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


# ── RAG Pre-flight Analysis ───────────────────────
import re as _re

_FUNC_RE = _re.compile(
    r"(?:def|function|const|async function|export function)\s+(\w+)",
    _re.IGNORECASE,
)


async def analyze_rag_results(task: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {"has_relevant_code": False, "summary": "", "recommendation": "implement_new", "existing_functions": []}

    code_text = " ".join(c["content"] for c in chunks).lower()
    task_keywords = [w for w in task.lower().split() if len(w) > 3]
    matches = [kw for kw in task_keywords if kw in code_text]
    has_relevant = len(matches) >= 2

    found_funcs = []
    for c in chunks:
        found_funcs.extend(_FUNC_RE.findall(c["content"]))
    found_funcs = list(set(found_funcs))[:5]

    if has_relevant and found_funcs:
        recommendation = "extend_existing"
    elif has_relevant:
        recommendation = "use_existing"
    else:
        recommendation = "implement_new"

    return {
        "has_relevant_code": has_relevant,
        "existing_functions": found_funcs,
        "summary": f"Found {len(found_funcs)} functions in {len(chunks)} chunks" if has_relevant else "No relevant code found",
        "recommendation": recommendation,
        "matched_keywords": matches,
    }


def build_prompt_with_analysis(task: str, chunks: list[dict], analysis: dict) -> str:
    NL = chr(10)
    ctx_lines = []
    for c in chunks:
        header = "### " + c.get("file", "") + " (relevance: " + str(c.get("score", 0)) + ")"
        ctx_lines.append(header)
        ctx_lines.append(c.get("content", ""))
    ctx = (NL + NL).join(ctx_lines)

    rec      = analysis.get("recommendation", "implement_new")
    existing = analysis.get("existing_functions", [])
    summary  = analysis.get("summary", "")

    if rec == "use_existing":
        directive = "Existing code: " + summary + ". Functions: " + str(existing) + ". Reuse or extend."
    elif rec == "extend_existing":
        directive = "Partial match: " + summary + ". Existing: " + str(existing) + ". Extend, dont rewrite."
    else:
        directive = "No existing implementation. Create new code fitting the codebase style."

    parts = [
        "Task: " + task,
        "Analysis: " + directive,
        "Relevant code context:",
        ctx,
        "Return JSON diff:",
        ]
    return (NL + NL).join(parts)



# ── Model Router ──────────────────────────────────
SYSTEM_PROMPT = """You are a code editing assistant.
Return ONLY a JSON object with this exact schema:
{
  "file": "<relative file path>",
  "changes": [
    {
      "type": "replace",
      "start_line": <int>,
      "end_line": <int>,
      "new_content": "<new code here>"
    }
  ]
}

Rules:
- For NEW files: start_line=1, end_line=0, new_content=full file content
- For EDITING existing code: use exact line numbers from the context provided
- For APPENDING to a file: use start_line=last_line+1, end_line=last_line
- No explanation. No markdown fences. Raw JSON only."""


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


async def get_openai_codex_access_token() -> str:
    refresh_token = await get_config("openai_codex_refresh_token")
    if not refresh_token:
        raise HTTPException(500, detail="OpenAI Codex OAuth is not configured. Connect it in bootstrap first.")

    expires_at_raw = await get_config("openai_codex_expires_at", "0")
    access_token = await get_config("openai_codex_access_token")
    try:
        expires_at = int(expires_at_raw or "0")
    except ValueError:
        expires_at = 0

    if access_token and expires_at > int(time.time()) + 15:
        return access_token

    try:
        token_data = await refresh_openai_codex_token(refresh_token=refresh_token)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response is not None else str(exc)
        raise HTTPException(401, detail=f"OpenAI Codex session expired or was revoked. Reconnect OAuth in bootstrap. {detail}")
    except httpx.HTTPError as exc:
        raise HTTPException(502, detail=f"Failed to refresh OpenAI Codex token: {exc}")

    await set_config("openai_codex_access_token", token_data["access_token"])
    await set_config("openai_codex_refresh_token", token_data.get("refresh_token") or refresh_token)
    await set_config("openai_codex_expires_at", str(token_data["expires_at"]))
    await set_config("openai_codex_account_id", token_data.get("account_id", ""))
    await set_config("openai_codex_token_type", token_data.get("token_type", "Bearer"))
    await set_config("openai_codex_scope", token_data.get("scope", ""))
    return token_data["access_token"]


async def call_model(task: str, chunks: list[dict], model: str, analysis: dict = None) -> tuple[dict, dict]:
    """Credentials đọc từ SQLite mỗi call. Analysis context từ RAG pre-flight."""
    provider, provider_model = normalize_model_name(model)
    # Dùng analysis-aware prompt nếu có
    if analysis:
        prompt = build_prompt_with_analysis(task, chunks, analysis)
    else:
        prompt = build_prompt(task, chunks)
    metrics  = {"tokens": {"prompt": 0, "completion": 0, "total": 0}, "provider": provider.value}

    # ── OpenAI ──────────────────────────────────
    if provider == ModelProvider.OPENAI:
        api_key = await get_config("openai_key")
        if not api_key:
            raise HTTPException(500, detail="OpenAI not configured. Re-run bootstrap.")
        client = OpenAI(api_key=api_key)
        try:
            resp = client.chat.completions.create(
                model=provider_model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                max_tokens=2000,
            )
        except RateLimitError as e:
            raise HTTPException(429, detail=f"OpenAI quota/rate limit for model '{model}': {e}")
        except AuthenticationError as e:
            raise HTTPException(401, detail=f"OpenAI authentication failed: {e}")
        except PermissionDeniedError as e:
            raise HTTPException(403, detail=f"OpenAI permission denied for model '{model}': {e}")
        except NotFoundError as e:
            raise HTTPException(404, detail=f"OpenAI model '{model}' not found or not enabled for this key: {e}")
        except APIError as e:
            raise HTTPException(502, detail=f"OpenAI API error for model '{model}': {e}")
        if resp.usage:
            metrics["tokens"] = {"prompt": resp.usage.prompt_tokens,
                                 "completion": resp.usage.completion_tokens,
                                 "total": resp.usage.total_tokens}
        return parse_json(resp.choices[0].message.content, model), metrics

    # ── OpenAI Codex OAuth ───────────────────────
    if provider == ModelProvider.OPENAI_CODEX:
        access_token = await get_openai_codex_access_token()
        client = OpenAI(api_key=access_token, base_url=OPENAI_CODEX_API_BASE)
        try:
            resp = client.chat.completions.create(
                model=provider_model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                max_tokens=2000,
            )
        except RateLimitError as e:
            raise HTTPException(429, detail=f"OpenAI Codex rate limit for model '{model}': {e}")
        except AuthenticationError as e:
            raise HTTPException(401, detail=f"OpenAI Codex authentication failed. Reconnect OAuth in bootstrap. {e}")
        except PermissionDeniedError as e:
            raise HTTPException(403, detail=f"OpenAI Codex model '{model}' is not available for this account: {e}")
        except NotFoundError as e:
            raise HTTPException(404, detail=f"OpenAI Codex model '{model}' was not found for OAuth access: {e}")
        except APIError as e:
            raise HTTPException(502, detail=f"OpenAI Codex API error for model '{model}': {e}")
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
        client = OpenAI(api_key=token, base_url=f"{copilot_url}/v1")
        try:
            resp = client.chat.completions.create(
                model=provider_model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                max_tokens=2000,
            )
        except APIError as e:
            raise HTTPException(502, detail=f"Copilot backend error via {copilot_url}: {e}")
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

    # File mới — model muốn tạo file, cho pass
    if not path.exists():
        return True

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    total = len(lines)

    for change in diff.get("changes", []):
        start = change.get("start_line", 0)
        end   = change.get("end_line", 0)

        # start_line = 0 → append vào cuối file, hợp lệ
        if start == 0 and end == 0:
            continue

        # Nếu line range vượt quá file nhưng start hợp lệ → model muốn append, cho pass
        if start <= total + 1:
            continue

        # start vượt quá file hoàn toàn → invalid
        return False

    return True


# ── Routes ────────────────────────────────────────
@router.post("/index/cancel")
async def cancel_index(body: dict):
    """UI gọi để cancel indexing đang chạy."""
    import asyncio
    task_id = body.get("task_id")
    if task_id and task_id in _cancel_flags:
        _cancel_flags[task_id].set()
        return {"status": "cancelled", "task_id": task_id}
    return {"status": "not_found"}


@router.post("/task")
async def handle_task(req: TaskRequest):
    """Non-streaming fallback — kept for OpenClaw direct calls."""
    project_path = await resolve_project_path(req.project)
    model = req.model or await get_config("default_model", "gpt-4.1")
    timings = {}
    t0 = time.perf_counter()

    t_rag = time.perf_counter()
    chunks = await query_rag(req.task, req.project, req.top_k)
    timings["rag_ms"] = round((time.perf_counter() - t_rag) * 1000)
    if not chunks:
        raise HTTPException(404, detail=f"No code found in '{req.project}'. Run /index/full?project={req.project} first.")

    analysis = await analyze_rag_results(req.task, chunks)

    t_model = time.perf_counter()
    diff, token_metrics = await call_model(req.task, chunks, model, analysis)
    timings["model_ms"] = round((time.perf_counter() - t_model) * 1000)

    if not await validate_diff(diff, project_path):
        raise HTTPException(409, detail="Diff validation failed — file may have changed. Re-index and retry.")

    timings["total_ms"] = round((time.perf_counter() - t0) * 1000)
    return {
        "diff": diff, "project": req.project,
        "analysis": {"recommendation": analysis.get("recommendation"),
                     "existing_functions": analysis.get("existing_functions", []),
                     "summary": analysis.get("summary")},
        "metrics": {"model": model, "provider": token_metrics["provider"],
                    "tokens": token_metrics["tokens"], "timings": timings,
                    "rag_sources": [{"file": c["file"], "score": c["score"]} for c in chunks],
                    "rag_chunks": len(chunks)},
    }


@router.get("/task/stream")
async def handle_task_stream(
        task: str,
        project: str,
        model: str = "",
        top_k: int = 5,
):
    """SSE streaming task — UI dùng endpoint này để show thinking steps."""
    from fastapi.responses import StreamingResponse
    import json as _json

    SEP = chr(10) + chr(10)

    def evt(type_: str, **kwargs) -> str:
        return "data: " + _json.dumps({"type": type_, **kwargs}) + SEP

    async def stream():
        try:
            # Step 1: resolve project
            yield evt("step", step="project", status="running", msg="Resolving project path...")
            project_path = await resolve_project_path(project)
            yield evt("step", step="project", status="done", msg="Project: " + str(project_path))

            # Step 2: resolve model
            resolved_model = model or await get_config("default_model", "gpt-4.1")
            yield evt("step", step="model", status="done", msg="Model: " + resolved_model)

            # Step 3: Ollama embedding
            yield evt("step", step="embedding", status="running",
                      msg="Ollama generating embeddings for task...")
            t_rag = time.perf_counter()
            chunks = await query_rag(task, project, top_k)
            rag_ms = round((time.perf_counter() - t_rag) * 1000)

            if not chunks:
                yield evt("error", msg="No code found in '" + project + "'. Run /index first.")
                return

            yield evt("step", step="embedding", status="done",
                      msg="Embedding done in " + str(rag_ms) + "ms — " + str(len(chunks)) + " chunks found",
                      rag_ms=rag_ms)

            # Step 4: show RAG sources
            sources = [{"file": c["file"], "score": c["score"]} for c in chunks]
            yield evt("step", step="rag", status="done",
                      msg="RAG sources found:",
                      sources=sources)

            # Step 5: analyze existing code
            yield evt("step", step="analysis", status="running",
                      msg="Analyzing existing codebase...")
            analysis = await analyze_rag_results(task, chunks)
            rec = analysis.get("recommendation", "implement_new")
            funcs = analysis.get("existing_functions", [])
            summary = analysis.get("summary", "")
            yield evt("step", step="analysis", status="done",
                      msg=summary,
                      recommendation=rec,
                      existing_functions=funcs)

            # Step 6: call model
            yield evt("step", step="model_call", status="running",
                      msg="Calling " + resolved_model + "...")
            t_model = time.perf_counter()
            diff, token_metrics = await call_model(task, chunks, resolved_model, analysis)
            model_ms = round((time.perf_counter() - t_model) * 1000)
            yield evt("step", step="model_call", status="done",
                      msg="Response in " + str(model_ms) + "ms — " + str(token_metrics["tokens"].get("total", 0)) + " tokens",
                      model_ms=model_ms,
                      tokens=token_metrics["tokens"])

            # Step 7: validate diff
            yield evt("step", step="validate", status="running", msg="Validating diff...")
            valid = await validate_diff(diff, project_path)
            if not valid:
                yield evt("error", msg="Diff validation failed — file may have changed. Re-index and retry.")
                return
            yield evt("step", step="validate", status="done", msg="Diff looks good")

            # Done
            yield evt("done",
                      diff=diff,
                      project=project,
                      analysis={"recommendation": rec,
                                "existing_functions": funcs,
                                "summary": summary},
                      metrics={"model": resolved_model,
                               "provider": token_metrics["provider"],
                               "tokens": token_metrics["tokens"],
                               "timings": {"rag_ms": rag_ms, "model_ms": model_ms,
                                           "total_ms": rag_ms + model_ms},
                               "rag_sources": sources,
                               "rag_chunks": len(chunks)})

        except HTTPException as e:
            yield evt("error", msg=e.detail)
        except Exception as e:
            yield evt("error", msg=str(e))

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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


IGNORE_DIRS = {
    "node_modules", ".git", ".svn", ".hg",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "venv", ".venv", "env", ".env",
    "dist", "build", ".next", ".nuxt", ".output",
    "coverage", ".nyc_output",
    ".idea", ".vscode",
    "vendor", "target", "Pods",
    ".gradle", ".m2",
    "elm-stuff", "__MACOSX",
}

def _should_skip(rel: Path) -> bool:
    return any(part in IGNORE_DIRS for part in rel.parts)


@router.post("/index/full")
async def index_full(project: str, extensions: str = ".py,.ts,.js,.go,.java,.rs,.tsx,.jsx"):
    project_path = await resolve_project_path(project)
    indexed, errors = [], []
    for ext in extensions.split(","):
        for f in project_path.rglob(f"*{ext}"):
            rel_path = f.relative_to(project_path)
            if _should_skip(rel_path):
                continue
            rel = str(rel_path)
            try:
                await index_file(project, rel, project_path)
                indexed.append(rel)
            except Exception as e:
                errors.append({"file": rel, "error": str(e)})
    await update_project_indexed(project, len(indexed))
    return {"project": project, "indexed": len(indexed), "errors": errors}


@router.get("/index/stream")
async def index_full_stream(project: str, extensions: str = ".py,.ts,.js,.go,.java,.rs,.tsx,.jsx"):
    """SSE endpoint — stream indexing progress về browser."""
    from fastapi.responses import StreamingResponse
    import json as _json
    import time as _time

    project_path = await resolve_project_path(project)

    # Collect files trước để biết total
    ext_list = extensions.split(",")
    all_files = []
    for ext in ext_list:
        all_files.extend(project_path.rglob(f"*{ext}"))
    total = len(all_files)

    async def event_stream():
        indexed, errors = [], []
        t0 = _time.perf_counter()

        SEP = chr(10) + chr(10)
        msg = _json.dumps({"type": "start", "total": total, "project": project})
        yield "data: " + msg + SEP

        for i, f in enumerate(all_files):
            rel = str(f.relative_to(project_path))
            try:
                chunks = await index_file(project, rel, project_path)
                indexed.append(rel)
                status = "ok"
                err = None
            except Exception as e:
                errors.append({"file": rel, "error": str(e)})
                status = "error"
                err = str(e)
                chunks = 0

            elapsed   = _time.perf_counter() - t0
            done      = i + 1
            pct       = round(done / total * 100) if total else 100
            avg_ms    = round(elapsed / done * 1000)
            remaining = round((total - done) * avg_ms / 1000)

            msg = _json.dumps({
                "type": "file", "file": rel, "status": status,
                "error": err, "chunks": chunks, "done": done,
                "total": total, "pct": pct,
                "elapsed_ms": round(elapsed * 1000),
                "remaining_s": remaining,
            })
            yield "data: " + msg + SEP

        await update_project_indexed(project, len(indexed))
        total_ms = round((_time.perf_counter() - t0) * 1000)
        msg = _json.dumps({
            "type": "done", "indexed": len(indexed),
            "errors": len(errors), "total_ms": total_ms, "project": project,
        })
        yield "data: " + msg + SEP

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/services/health")
async def services_health():
    """Check Ollama + ChromaDB status."""
    import asyncio

    async def check_ollama():
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags", timeout=3)
                models = [m["name"] for m in r.json().get("models", [])]
                embed_model = await get_config("embed_model", "nomic-embed-text")
                has_model = any(embed_model in m for m in models)
                return {"ok": True, "models": models, "embed_ready": has_model}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def check_chroma():
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"http://{CHROMA_HOST}:{CHROMA_PORT}/api/v1/heartbeat", timeout=3)
                return {"ok": r.status_code == 200}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    ollama, chroma = await asyncio.gather(check_ollama(), check_chroma())
    return {
        "ollama": ollama,
        "chromadb": chroma,
        "ready": ollama["ok"] and ollama.get("embed_ready") and chroma["ok"],
    }


@router.get("/projects")
async def get_projects():
    projects = await db_list_projects()
    return {"projects": [p["name"] for p in projects]}


@router.get("/models")
async def get_models():
    default = await get_config("default_model", "")

    # Merge static registry + default model (có thể là model dynamic từ bootstrap)
    all_models = dict(MODEL_REGISTRY)
    if default and default not in all_models:
        try:
            provider = resolve_provider(default)
            all_models[default] = provider
        except Exception:
            pass

    return {
        "models":  list(all_models.keys()),
        "default": default,
        "by_provider": {
            p.value: [m for m, pv in all_models.items() if pv == p]
            for p in ModelProvider
        },
    }
