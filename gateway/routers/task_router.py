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

_cancel_flags: dict = {}
_active_streams: dict = {}  # task_id → {project, started_at, done, total, pct, status}

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
    OPENAI    = "openai"
    ANTHROPIC = "anthropic"
    GEMINI    = "gemini"
    CODEX     = "codex"
    COPILOT   = "copilot"  # via copilot-api sidecar

# Static registry — fallback và well-known models
MODEL_REGISTRY: dict[str, ModelProvider] = {
    # OpenAI Codex (ChatGPT OAuth)
    "openai-codex/gpt-5.4":            ModelProvider.CODEX,
    "openai-codex/gpt-5.3-codex":      ModelProvider.CODEX,
    "openai-codex/gpt-5.1-codex-mini": ModelProvider.CODEX,
    # GitHub Copilot (via copilot-api sidecar)
    "copilot/gpt-5.4":                 ModelProvider.COPILOT,
    "copilot/gpt-5.4-mini":            ModelProvider.COPILOT,
    "copilot/gpt-5.3-codex":           ModelProvider.COPILOT,
    "copilot/gpt-5.2-codex":           ModelProvider.COPILOT,
    "copilot/gpt-5.1-codex":           ModelProvider.COPILOT,
    "copilot/gpt-5.1-codex-mini":      ModelProvider.COPILOT,
    "copilot/gpt-4.1":                 ModelProvider.COPILOT,
    "copilot/gpt-4o":                  ModelProvider.COPILOT,
    "copilot/claude-sonnet-4.6":       ModelProvider.COPILOT,
    "copilot/claude-opus-4.6":         ModelProvider.COPILOT,
    "copilot/claude-sonnet-4.5":       ModelProvider.COPILOT,
    "copilot/claude-haiku-4.5":        ModelProvider.COPILOT,
    "copilot/gemini-2.5-pro":          ModelProvider.COPILOT,
    "copilot/gemini-3.1-pro-preview":  ModelProvider.COPILOT,
    "copilot/grok-code-fast-1":        ModelProvider.COPILOT,
    # OpenAI
    "gpt-4.1":              ModelProvider.OPENAI,
    "gpt-4.1-mini":         ModelProvider.OPENAI,
    "gpt-4o":               ModelProvider.OPENAI,
    "gpt-4o-mini":          ModelProvider.OPENAI,
    "gpt-5.4":              ModelProvider.OPENAI,
    "codex-mini-latest":    ModelProvider.OPENAI,
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
}


def resolve_provider(model: str) -> ModelProvider:
    """
    Resolve provider từ model name.
    Static registry trước, sau đó prefix matching cho dynamic models.
    """
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]

    # Prefix matching cho models không có trong registry
    m = model.lower()
    if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return ModelProvider.OPENAI
    if m.startswith("claude-"):
        return ModelProvider.ANTHROPIC
    if m.startswith("gemini-"):
        return ModelProvider.GEMINI
    raise HTTPException(400, detail=f"Unknown model '{model}'. Cannot determine provider.")


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
async def get_embedding(text: str, retries: int = 3) -> list[float]:
    embed_model = await get_config("embed_model", "nomic-embed-text")
    import asyncio as _aio
    last_err = None
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
                    json={"model": embed_model, "prompt": text},
                    timeout=300,  # 5 phút — Ollama CPU có thể chậm
                )
                resp.raise_for_status()
                return resp.json()["embedding"]
        except httpx.ConnectError:
            raise HTTPException(503, detail="Ollama is not running.")
        except (httpx.TimeoutException, httpx.RemoteProtocolError) as e:
            last_err = e
            if attempt < retries - 1:
                await _aio.sleep(2 ** attempt)  # backoff: 1s, 2s
                continue
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                await _aio.sleep(1)
                continue
    raise HTTPException(503, detail=f"Ollama embedding failed after {retries} attempts: {last_err}")


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


def _sync_call_model(task, chunks, model, analysis=None):
    """Sync wrapper — chạy trong executor để không block event loop."""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(call_model(task, chunks, model, analysis))
    finally:
        loop.close()


async def call_model(task: str, chunks: list[dict], model: str, analysis: dict = None) -> tuple[dict, dict]:
    """API key đọc từ SQLite mỗi call. Analysis context từ RAG pre-flight."""
    provider = resolve_provider(model)
    # Dùng analysis-aware prompt nếu có
    if analysis:
        prompt = build_prompt_with_analysis(task, chunks, analysis)
    else:
        prompt = build_prompt(task, chunks)
    metrics  = {"tokens": {"prompt": 0, "completion": 0, "total": 0}, "provider": provider.value}

    # ── OpenAI ──────────────────────────────────
    if provider == ModelProvider.OPENAI:
        # Prefer Codex OAuth token if available and valid
        import time as _t
        codex_access  = await get_config("openai_codex_access", "")
        codex_expires = int(await get_config("openai_codex_expires", "0"))
        codex_valid   = bool(codex_access) and codex_expires > int(_t.time())

        if codex_valid:
            api_key = codex_access
            base_url = "https://chatgpt.com/backend-api"
        else:
            api_key = await get_config("openai_key")
            base_url = None

        if not api_key:
            raise HTTPException(500, detail="OpenAI not configured. Re-run bootstrap.")
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
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

    # ── OpenAI Codex (ChatGPT OAuth) ─────────────
    if provider == ModelProvider.CODEX:
        import time as _t
        access  = await get_config("openai_codex_access", "")
        expires = int(await get_config("openai_codex_expires", "0"))
        if not access or expires <= int(_t.time()):
            raise HTTPException(500, detail="OpenAI Codex OAuth token expired. Re-authenticate in bootstrap.")
        from openai import OpenAI

        # Use chatgpt.com backend for Codex models
        codex_model = model.replace("openai-codex/", "")
        client = OpenAI(
            api_key=access,
            base_url="https://chatgpt.com/backend-api",
        )
        resp = client.chat.completions.create(
            model=codex_model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        if resp.usage:
            metrics["tokens"] = {"prompt": resp.usage.prompt_tokens,
                                 "completion": resp.usage.completion_tokens,
                                 "total": resp.usage.total_tokens}
        return parse_json(resp.choices[0].message.content, model), metrics

    # ── GitHub Copilot (via copilot-api sidecar) ────
    if provider == ModelProvider.COPILOT:
        copilot_url = os.getenv("COPILOT_API_URL", "http://copilot-api:4141")
        # Dùng GitHub token làm API key cho copilot-api
        github_token = await get_config("copilot_github_token", "copilot")
        from openai import OpenAI
        client = OpenAI(
            api_key=github_token,
            base_url=f"{copilot_url}/v1",
        )
        copilot_model = model.replace("copilot/", "")
        resp = client.chat.completions.create(
            model=copilot_model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",   "content": prompt}],
            max_tokens=2000,
        )
        if resp.usage:
            metrics["tokens"] = {"prompt":     resp.usage.prompt_tokens,
                                 "completion": resp.usage.completion_tokens,
                                 "total":      resp.usage.total_tokens}
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

    # ── Gemini (google-genai new SDK) ────────────────
    if provider == ModelProvider.GEMINI:
        api_key = await get_config("gemini_key")
        if not api_key:
            raise HTTPException(500, detail="Gemini not configured. Re-run bootstrap.")
        from google import genai as google_genai
        from google.genai import types as genai_types
        client = google_genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=2000,
            ),
        )
        usage = resp.usage_metadata
        if usage:
            metrics["tokens"] = {
                "prompt":     usage.prompt_token_count or 0,
                "completion": usage.candidates_token_count or 0,
                "total":      usage.total_token_count or 0,
            }
        return parse_json(resp.text, model), metrics

    raise HTTPException(500, detail=f"Unhandled provider: {provider}")


# ── Diff validation ───────────────────────────────
async def validate_diff(diff: dict, project_path: Path) -> bool:
    """
    Validate diff từ model.
    Intentionally lenient — model không biết chính xác line numbers,
    chỉ reject khi file path rõ ràng sai.
    """
    file_path = diff.get("file", "")
    if not file_path:
        return False

    path = project_path / file_path

    # File chưa tồn tại → model muốn tạo mới, pass
    if not path.exists():
        return True

    # File tồn tại → chỉ check file readable, không check line numbers
    # Model có thể trả line numbers không chính xác do RAG chunking
    # OpenClaw sẽ apply diff và handle nếu lệch
    try:
        path.read_text(encoding="utf-8", errors="ignore")
        return True
    except Exception:
        return False


# ── Routes ────────────────────────────────────────
@router.post("/index/cancel")
async def cancel_index(body: dict):
    """UI gọi để cancel indexing đang chạy."""
    task_id = body.get("task_id")
    if task_id and task_id in _cancel_flags:
        _cancel_flags[task_id].set()
        return {"status": "cancelled", "task_id": task_id}
    # Cancel tất cả nếu không có task_id
    if not task_id:
        for ev in _cancel_flags.values():
            ev.set()
        return {"status": "cancelled_all", "count": len(_cancel_flags)}
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
            import asyncio as _aio
            diff, token_metrics = await _aio.get_event_loop().run_in_executor(
                None, lambda: _sync_call_model(task, chunks, resolved_model, analysis)
            )
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

            # Save to history
            from db import save_chat_message
            await save_chat_message(project, "user", task)
            await save_chat_message(project, "assistant",
                                    diff.get("file", ""),
                                    {"recommendation": rec, "model": resolved_model,
                                     "tokens": token_metrics.get("tokens", {}),
                                     "timings": {"rag_ms": rag_ms, "model_ms": model_ms}})

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
    return any(part in IGNORE_DIRS for part in Path(rel).parts)



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
    """SSE endpoint with cancel support and IGNORE_DIRS filtering."""
    from fastapi.responses import StreamingResponse
    import json as _json
    import time as _time
    import asyncio as _asyncio
    import uuid as _uuid

    project_path = await resolve_project_path(project)
    ext_list     = extensions.split(",")
    SEP          = chr(10) + chr(10)

    task_id      = _uuid.uuid4().hex
    cancel_event = _asyncio.Event()
    _cancel_flags[task_id] = cancel_event
    _active_streams[task_id] = {
        "project": project, "status": "scanning",
        "done": 0, "total": 0, "pct": 0, "errors": 0,
        "started_at": _time.time(),
    }

    async def event_stream():
        indexed, errors = [], []
        t0 = _time.perf_counter()

        try:
            # Phase 1: scan + filter
            msg = _json.dumps({"type": "scanning", "project": project, "task_id": task_id})
            yield "data: " + msg + SEP

            # Scan trong thread + stream progress qua queue
            import threading as _threading
            scan_queue  = _asyncio.Queue()
            scan_count  = [0]  # mutable counter visible từ cả thread lẫn coroutine

            def scan_files():
                """
                Dùng os.walk với topdown=True để prune IGNORE_DIRS sớm.
                Nhanh hơn rglob vì không đi vào node_modules, dist, etc.
                """
                import os as _os
                found = []
                ext_set = set(ext_list)

                for root, dirs, files in _os.walk(str(project_path), topdown=True):
                    if cancel_event.is_set():
                        break

                    # Prune ignored dirs IN-PLACE — os.walk sẽ không đi vào
                    dirs[:] = [
                        d for d in dirs
                        if d not in IGNORE_DIRS and not d.startswith('.')
                           or d in {'.well-known'}  # exceptions
                    ]

                    root_path = Path(root)
                    for fname in files:
                        if any(fname.endswith(ext) for ext in ext_set):
                            found.append(root_path / fname)
                            scan_count[0] = len(found)
                            if len(found) % 50 == 0:
                                try:
                                    scan_queue.put_nowait(len(found))
                                except Exception:
                                    pass

                try:
                    scan_queue.put_nowait(None)
                except Exception:
                    pass
                return found

            loop = _asyncio.get_event_loop()
            scan_future = loop.run_in_executor(None, scan_files)

            # Stream progress while thread is scanning
            while True:
                try:
                    item = await _asyncio.wait_for(
                        _asyncio.shield(scan_queue.get()), timeout=0.3
                    )
                    if item is None:
                        break
                    msg = _json.dumps({"type": "scanning", "found": item, "task_id": task_id})
                    yield "data: " + msg + SEP
                except _asyncio.TimeoutError:
                    # Heartbeat dùng counter — luôn có số thật
                    current = scan_count[0]
                    msg = _json.dumps({"type": "scanning", "found": current, "task_id": task_id})
                    yield "data: " + msg + SEP

            all_files = await scan_future

            if cancel_event.is_set():
                msg = _json.dumps({"type": "cancelled", "project": project, "scanned": len(all_files)})
                yield "data: " + msg + SEP
                return

            msg = _json.dumps({"type": "scanning", "found": len(all_files), "task_id": task_id})
            yield "data: " + msg + SEP

            total = len(all_files)
            msg = _json.dumps({"type": "start", "total": total, "project": project, "task_id": task_id})
            yield "data: " + msg + SEP

            # Phase 2: index in concurrent batches
            # Sequential files — Ollama single-threaded, batch chỉ gây timeout
            BATCH = 1
            done_count = 0

            # index_one defined OUTSIDE loop to avoid closure bug
            async def index_one(f):
                rel = str(f.relative_to(project_path))
                try:
                    c = await index_file(project, rel, project_path)
                    return {"rel": rel, "chunks": c, "status": "ok", "err": None}
                except Exception as e:
                    return {"rel": rel, "chunks": 0, "status": "error", "err": str(e)}

            for batch_start in range(0, total, BATCH):
                if cancel_event.is_set():
                    msg = _json.dumps({"type": "cancelled", "project": project,
                                       "done": done_count, "total": total})
                    yield "data: " + msg + SEP
                    return

                batch   = all_files[batch_start:batch_start + BATCH]
                results = await _asyncio.gather(*[index_one(f) for f in batch])

                for r in results:
                    done_count += 1
                    if r["status"] == "ok":
                        indexed.append(r["rel"])
                    else:
                        errors.append({"file": r["rel"], "error": r["err"]})

                    # Update dashboard registry
                    if task_id in _active_streams:
                        stream_data = {
                            "status": "indexing",
                            "done": done_count, "total": total,
                            "pct": round(done_count / total * 100) if total else 0,
                            "errors": len(errors),
                            "current_file": r["rel"],
                        }
                        if r["status"] == "error":
                            # Keep last 50 errors for dashboard
                            errs = _active_streams[task_id].get("errors_detail", [])
                            errs.append({"file": r["rel"], "error": r["err"]})
                            stream_data["errors_detail"] = errs[-50:]
                        else:
                            stream_data["errors_detail"] = _active_streams[task_id].get("errors_detail", [])
                        _active_streams[task_id].update(stream_data)

                    elapsed   = _time.perf_counter() - t0
                    pct       = round(done_count / total * 100) if total else 100
                    avg_ms    = round(elapsed / done_count * 1000)
                    remaining = round((total - done_count) * avg_ms / 1000)

                    msg = _json.dumps({
                        "type": "file", "file": r["rel"], "status": r["status"],
                        "error": r["err"], "chunks": r["chunks"],
                        "done": done_count, "total": total, "pct": pct,
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

        finally:
            _cancel_flags.pop(task_id, None)
            _active_streams.pop(task_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Active index streams (for dashboard) ─────────
@router.get("/index/active")
async def list_active_streams():
    """List all currently running index streams."""
    import time as _t
    return {
        "streams": [
            {"task_id": tid, **info, "elapsed_s": round(_t.time() - info["started_at"])}
            for tid, info in _active_streams.items()
        ]
    }


# ── Services health ───────────────────────────────
@router.get("/services/health")
async def services_health():
    """Check Ollama + ChromaDB + Copilot status."""
    import asyncio as _aio

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

    async def check_copilot():
        copilot_url = os.getenv("COPILOT_API_URL", "http://copilot-api:4141")
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{copilot_url}/v1/models", timeout=5)
            if r.status_code == 200:
                models = [m["id"] for m in r.json().get("data", [])]
                return {"ok": True, "authed": True, "models": models}
            return {"ok": True, "authed": False, "error": f"HTTP {r.status_code}"}
        except httpx.ConnectError:
            return {"ok": False, "authed": False, "error": "not running"}
        except Exception as e:
            return {"ok": False, "authed": False, "error": str(e)}

    ollama, chroma, copilot = await _aio.gather(
        check_ollama(), check_chroma(), check_copilot()
    )
    return {
        "ollama":   ollama,
        "chromadb": chroma,
        "copilot":  copilot,
        "ready":    ollama["ok"] and ollama.get("embed_ready") and chroma["ok"],
    }


# ── Models ────────────────────────────────────────
@router.get("/models")
async def get_models():
    default = await get_config("default_model", "")
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


# ── Projects ──────────────────────────────────────
@router.get("/projects")
async def get_projects():
    projects = await db_list_projects()
    return {"projects": [p["name"] for p in projects]}


# ── Chat history ──────────────────────────────────
@router.get("/history/{project}")
async def get_history(project: str, limit: int = 50):
    from db import get_chat_history
    history = await get_chat_history(project, limit)
    return {"project": project, "history": history}


@router.delete("/history/{project}")
async def clear_history(project: str):
    from db import clear_chat_history
    await clear_chat_history(project)
    return {"status": "cleared", "project": project}


@router.post("/history/{project}/approve")
async def approve_diff(project: str, body: dict):
    """Apply diff to file then re-index."""
    file_path = body.get("file_path", "")
    changes   = body.get("changes", [])

    if not file_path:
        raise HTTPException(400, detail="file_path required")

    project_path = await resolve_project_path(project)
    full_path    = project_path / file_path

    if changes:
        if full_path.exists():
            lines = full_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        else:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []

        for change in sorted(changes, key=lambda c: c.get("start_line", 0), reverse=True):
            start    = change.get("start_line", 1)
            end      = change.get("end_line", 0)
            new_code = change.get("new_content", "")
            new_lines = new_code.splitlines()

            if start == 1 and end == 0:
                lines = new_lines
            elif start > 0 and end >= start:
                lines = lines[:start-1] + new_lines + lines[end:]
            elif start > len(lines):
                lines.extend(new_lines)
            else:
                lines = lines[:max(0,start-1)] + new_lines + lines[max(0,start-1):]

        full_path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")

    chunks = await index_file(project, file_path, project_path)
    await update_project_indexed(project, chunks)

    from db import save_chat_message
    await save_chat_message(project, "system",
                            "Applied: " + file_path, {"action": "approve", "file": file_path})

    return {"status": "applied", "file": file_path, "chunks": chunks}
