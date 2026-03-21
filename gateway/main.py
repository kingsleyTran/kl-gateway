"""
OpenClaw Gateway — entrypoint.
Bootstrap state đọc từ SQLite mỗi request — không cache, không .env.
Dùng lifespan để init DB đúng cách với uvloop.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from db import init_db, is_bootstrapped

CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "/config"))
STATIC_DIR = Path(__file__).parent / "bootstrap" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — chạy trong event loop của uvicorn/uvloop, không conflict
    await init_db(CONFIG_DIR)
    yield
    # Shutdown (không cần làm gì)


app = FastAPI(title="OpenClaw Gateway", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Middleware ─────────────────────────────────────
ALWAYS_ALLOW = ("/api/bootstrap", "/health", "/static", "/", "/dashboard")

@app.middleware("http")
async def bootstrap_guard(request: Request, call_next):
    path = request.url.path
    if any(path.startswith(p) for p in ALWAYS_ALLOW):
        return await call_next(request)
    if not await is_bootstrapped():
        return JSONResponse(status_code=503, content={
            "error": "Gateway not configured",
            "message": "Complete setup at http://localhost:8080",
        })
    return await call_next(request)


# ── Routers ───────────────────────────────────────
from routers import bootstrap_router, task_router, project_router, jobs_router

app.include_router(bootstrap_router.router, prefix="/api/bootstrap", tags=["bootstrap"])
app.include_router(project_router.router,   prefix="/api/projects",  tags=["projects"])
app.include_router(task_router.router,                               tags=["tasks"])
app.include_router(jobs_router.router,                               tags=["jobs"])


# ── Root ──────────────────────────────────────────
@app.get("/")
async def root():
    if not await is_bootstrapped():
        return FileResponse(str(STATIC_DIR / "index.html"))
    return FileResponse(str(STATIC_DIR / "ready.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "bootstrapped": await is_bootstrapped()}


@app.get("/dashboard")
async def dashboard():
    return FileResponse(str(STATIC_DIR / "dashboard.html"))


@app.get("/skill.md")
async def serve_skill():
    skill = Path(__file__).parent.parent / "SKILL.md"
    return FileResponse(str(skill), media_type="text/markdown")
