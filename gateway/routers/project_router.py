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
# Filesystem browser endpoint
# ─────────────────────────────────────────
@router.get("/browse")
async def browse_filesystem(path: str = ""):
    """
    Browse folders trên host filesystem.
    Trả về children folders của path, hoặc home dir nếu path rỗng.
    """
    if not path:
        # Start từ /projects (mount point cố định)
        prefix = os.getenv("HOST_MOUNT_PREFIX", "/projects")
        real_path = Path("/")
        container_path = Path(prefix)
    else:
        # path từ UI là host path thật, prefix /host để ra container path
        real_path = path.rstrip("/")
        container_path = Path(HOST_PREFIX + real_path)

    if not container_path.exists() or not container_path.is_dir():
        raise HTTPException(400, detail=f"Invalid path: {real_path}")

    # Parent (tính từ container_path, strip prefix sau)
    parent = str(container_path.parent) if str(container_path) != str(container_path.parent) else None

    # List subdirectories — bỏ hidden folders
    children = []
    try:
        for item in sorted(container_path.iterdir()):
            if not item.is_dir():
                continue
            name = item.name
            if name.startswith("."):
                continue
            # Quick file count (non-recursive, chỉ để hint)
            try:
                has_code = any(
                    f.suffix in {".py",".ts",".js",".go",".java",".rs",".tsx",".jsx"}
                    for f in item.iterdir()
                    if f.is_file()
                )
            except PermissionError:
                has_code = False

            children.append({
                "name":     name,
                "path":     str(container_path / name),
                "has_code": has_code,
            })
    except PermissionError:
        raise HTTPException(403, detail=f"Permission denied: {real_path}")

    # Strip HOST_PREFIX (/host) khỏi paths trả về UI
    # /host/Users/kingsley/Desktop → /Users/kingsley/Desktop
    prefix = os.getenv("HOST_MOUNT_PREFIX", "")

    def strip_prefix(p: str) -> str:
        if prefix and p.startswith(prefix):
            stripped = p[len(prefix):]
            return stripped if stripped else "/"
        return p

    return {
        "current":  strip_prefix(str(container_path)),
        "parent":   strip_prefix(parent) if parent else None,
        "children": [
            {**c, "path": strip_prefix(c["path"])}
            for c in children
        ],
    }


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
    # Path từ UI là path thật của host (đã strip /host prefix)
    # Không dùng resolve() vì sẽ resolve trong context container
    real_path = body.path.rstrip("/")
    container_path = Path(HOST_PREFIX + real_path)

    if not container_path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {real_path}")
    if not container_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {real_path}")

    existing = await get_project(body.name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Project '{body.name}' already exists.")

    # Lưu path thật (không có /host prefix) vào DB
    project = await add_project(body.name, real_path, body.description)
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

    # Xóa ChromaDB collection
    try:
        import chromadb, os
        chroma = chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "chromadb"),
            port=int(os.getenv("CHROMA_PORT", 8000)),
        )
        chroma.delete_collection(name)
    except Exception:
        pass  # Collection chưa tồn tại thì bỏ qua

    await delete_project(name)
    return {"status": "deleted", "name": name}


@router.post("/validate-path")
async def validate_path(body: dict):
    """
    UI gọi để validate path real-time trước khi submit.
    """
    raw = body.get("path", "").rstrip("/")
    container_path = Path(HOST_PREFIX + raw)
    exists = container_path.exists() and container_path.is_dir()

    file_count = 0
    if exists:
        extensions = [".py", ".ts", ".js", ".go", ".java", ".rs", ".tsx", ".jsx"]
        for ext in extensions:
            file_count += len(list(container_path.rglob(f"*{ext}")))

    return {
        "valid":      exists,
        "resolved":   raw,
        "file_count": file_count,
    }
