"""
Project router — quản lý projects qua SQLite.
Path là filesystem path thật trên host, không cần mount Docker volume.
"""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db import list_projects, get_project, add_project, delete_project

router = APIRouter()

# Trong Docker: host filesystem được mount tại /host
# Native: prefix rỗng
HOST_PREFIX = os.getenv("HOST_MOUNT_PREFIX", "")

def host_path(p: str) -> Path:
    """Convert path thật của user → path trong container."""
    return Path(HOST_PREFIX + str(Path(p).expanduser().resolve()))


class AddProjectRequest(BaseModel):
    name: str
    path: str
    description: str = ""


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@router.get("")
async def get_projects():
    """List tất cả projects đã đăng ký."""
    projects = await list_projects()
    return {"projects": projects}


@router.post("")
async def create_project(body: AddProjectRequest):
    """
    Thêm project mới.
    Validate path tồn tại trên filesystem trước khi lưu.
    """
    real_path = Path(body.path).expanduser().resolve()
    container_path = host_path(str(real_path))

    if not container_path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {real_path}")
    if not container_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {real_path}")

    existing = await get_project(body.name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Project '{body.name}' already exists.")

    # Lưu path thật (không có prefix) vào DB
    project = await add_project(body.name, str(real_path), body.description)
    return {"status": "created", "project": project}


@router.get("/{name}")
async def get_project_detail(name: str):
    project = await get_project(name)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found.")

    # Thêm file count preview
    path = host_path(project["path"])
    extensions = [".py", ".ts", ".js", ".go", ".java", ".rs", ".tsx", ".jsx"]
    files = []
    for ext in extensions:
        files.extend(path.rglob(f"*{ext}"))

    return {**project, "file_preview": len(files)}


@router.delete("/{name}")
async def remove_project(name: str):
    project = await get_project(name)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found.")
    await delete_project(name)
    return {"status": "deleted", "name": name}


@router.post("/validate-path")
async def validate_path(body: dict):
    """
    UI gọi để validate path real-time trước khi submit.
    """
    raw = body.get("path", "")
    real_path = Path(raw).expanduser().resolve()
    container_path = host_path(str(real_path))
    exists = container_path.exists() and container_path.is_dir()

    file_count = 0
    if exists:
        extensions = [".py", ".ts", ".js", ".go", ".java", ".rs", ".tsx", ".jsx"]
        for ext in extensions:
            file_count += len(list(container_path.rglob(f"*{ext}")))

    return {
        "valid":      exists,
        "resolved":   str(real_path),
        "file_count": file_count,
    }
