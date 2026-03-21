"""
Index Worker — chạy bởi arq (Redis-based async job queue).
Xử lý indexing jobs với retry, cancel, và progress tracking.
"""

import os
import asyncio
import json
import time
from pathlib import Path

import chromadb
import httpx
from arq.connections import RedisSettings

REDIS_URL   = os.getenv("REDIS_URL", "redis://redis:6379")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
HOST_PREFIX = os.getenv("HOST_MOUNT_PREFIX", "/projects")

IGNORE_DIRS = {
    "node_modules", ".git", ".svn", ".hg",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "venv", ".venv", "env", ".env",
    "dist", "build", ".next", ".nuxt", ".output",
    "coverage", ".nyc_output", ".idea", ".vscode",
    "vendor", "target", "Pods", ".gradle", ".m2",
    "elm-stuff", "__MACOSX",
}

BATCH_SIZE   = 10
EMBED_CONCUR = 8


def progress_channel(job_id: str) -> str:
    return "index:progress:" + job_id


def should_skip(rel: Path) -> bool:
    return any(part in IGNORE_DIRS for part in rel.parts)


def get_project_path(project: str) -> Path:
    path = Path(HOST_PREFIX) / project
    if not path.exists():
        raise FileNotFoundError("Project not found: " + str(path))
    return path


async def get_embedding(text: str, client: httpx.AsyncClient) -> list:
    resp = await client.post(
        "http://" + OLLAMA_HOST + ":" + str(OLLAMA_PORT) + "/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
        )
    resp.raise_for_status()
    return resp.json()["embedding"]


async def index_one_file(project, file_path, project_path, chroma, http, sem):
    full = project_path / file_path
    if not full.exists():
        raise FileNotFoundError(file_path)

    text   = full.read_text(encoding="utf-8", errors="ignore")
    lines  = text.splitlines()
    chunks = ["\n".join(lines[i:i+100]) for i in range(0, len(lines), 100)]
    if not chunks:
        return 0

    async def embed_one(chunk):
        async with sem:
            return await get_embedding(chunk, http)

    embeddings = await asyncio.gather(*[embed_one(c) for c in chunks])

    col = chroma.get_or_create_collection(project, metadata={"hnsw:space": "cosine"})
    existing = col.get(where={"file": file_path})
    if existing["ids"]:
        col.delete(ids=existing["ids"])

    col.upsert(
        ids       =[file_path + "::chunk_" + str(i) for i in range(len(chunks))],
        embeddings=embeddings,
        documents =chunks,
        metadatas =[{"file": file_path, "chunk_idx": i} for i in range(len(chunks))],
    )
    return len(chunks)


async def index_project(ctx, project: str, extensions: str = ".py,.ts,.js,.go,.java,.rs,.tsx,.jsx"):
    """Main index job. Progress published to Redis pub/sub."""
    job_id = ctx["job_id"]
    r      = ctx["redis"]

    async def pub(data: dict):
        await r.publish(progress_channel(job_id), json.dumps(data))

    await pub({"type": "started", "job_id": job_id, "project": project})

    try:
        project_path = get_project_path(project)
    except FileNotFoundError as e:
        await pub({"type": "error", "job_id": job_id, "error": str(e)})
        raise

    chroma   = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    sem      = asyncio.Semaphore(EMBED_CONCUR)
    ext_list = extensions.split(",")
    t0       = time.perf_counter()

    # Phase 1: scan
    await pub({"type": "scanning", "job_id": job_id, "project": project})
    all_files = []
    for ext in ext_list:
        for f in project_path.rglob("*" + ext):
            rel = f.relative_to(project_path)
            if not should_skip(rel):
                all_files.append(f)

    total = len(all_files)
    await pub({"type": "scan_done", "job_id": job_id, "total": total})

    indexed, errors = [], []
    done_count = 0

    async with httpx.AsyncClient() as http:
        for batch_start in range(0, total, BATCH_SIZE):
            # Check cancel flag in Redis
            cancelled = await r.get("index:cancel:" + job_id)
            if cancelled:
                await pub({"type": "cancelled", "job_id": job_id, "done": done_count, "total": total})
                await r.delete("index:cancel:" + job_id)
                return {"status": "cancelled", "indexed": len(indexed)}

            batch = all_files[batch_start:batch_start + BATCH_SIZE]

            async def process_one(f):
                rel = str(f.relative_to(project_path))
                try:
                    chunks = await asyncio.wait_for(
                        index_one_file(project, rel, project_path, chroma, http, sem),
                        timeout=60.0,
                    )
                    return {"file": rel, "chunks": chunks, "status": "ok", "error": None}
                except asyncio.TimeoutError:
                    return {"file": rel, "chunks": 0, "status": "error", "error": "timeout"}
                except Exception as e:
                    return {"file": rel, "chunks": 0, "status": "error", "error": str(e)}

            results = await asyncio.gather(*[process_one(f) for f in batch])

            for res in results:
                done_count += 1
                if res["status"] == "ok":
                    indexed.append(res["file"])
                else:
                    errors.append(res)

                elapsed   = time.perf_counter() - t0
                pct       = round(done_count / total * 100) if total else 100
                avg_ms    = round(elapsed / done_count * 1000)
                remaining = round((total - done_count) * avg_ms / 1000)

                await pub({
                    "type": "file", "job_id": job_id,
                    "file": res["file"], "status": res["status"],
                    "error": res["error"], "chunks": res["chunks"],
                    "done": done_count, "total": total, "pct": pct,
                    "elapsed_ms": round(elapsed * 1000),
                    "remaining_s": remaining,
                })

    total_ms = round((time.perf_counter() - t0) * 1000)
    await pub({
        "type": "done", "job_id": job_id, "project": project,
        "indexed": len(indexed), "errors": len(errors), "total_ms": total_ms,
    })
    return {"indexed": len(indexed), "errors": len(errors), "total_ms": total_ms}


class WorkerSettings:
    functions      = [index_project]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    max_jobs       = 4
    job_timeout    = 3600
    max_tries      = 3
    keep_result    = 3600
