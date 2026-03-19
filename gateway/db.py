"""
Single SQLite DB cho toàn bộ gateway state.
Tables:
  - projects  : project registry
  - config    : key-value store cho API keys, model defaults, etc.
"""

import aiosqlite
from pathlib import Path

_DB_PATH: Path | None = None


def get_db_path() -> Path:
    if _DB_PATH is None:
        raise RuntimeError("db not initialized — call init_db() first")
    return _DB_PATH


async def init_db(config_dir: Path):
    global _DB_PATH
    config_dir.mkdir(parents=True, exist_ok=True)
    _DB_PATH = config_dir / "gateway.db"

    async with aiosqlite.connect(_DB_PATH) as db:
        # Projects
        await db.execute("""
                         CREATE TABLE IF NOT EXISTS projects (
                                                                 id          INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                 name        TEXT UNIQUE NOT NULL,
                                                                 path        TEXT NOT NULL,
                                                                 description TEXT DEFAULT '',
                                                                 indexed     INTEGER DEFAULT 0,
                                                                 file_count  INTEGER DEFAULT 0,
                                                                 created_at  TEXT DEFAULT (datetime('now'))
                             )
                         """)

        # Config — key/value store
        # Keys: openai_key, gemini_key, copilot_token,
        #       default_model, chroma_host, ollama_host, embed_model
        await db.execute("""
                         CREATE TABLE IF NOT EXISTS config (
                                                               key   TEXT PRIMARY KEY,
                                                               value TEXT NOT NULL
                         )
                         """)

        # Seed infra defaults nếu chưa có
        await db.execute("""
                         INSERT OR IGNORE INTO config (key, value) VALUES
                ('chroma_host',  'chromadb'),
                ('chroma_port',  '8000'),
                ('ollama_host',  'ollama'),
                ('ollama_port',  '11434'),
                ('embed_model',  'nomic-embed-text'),
                ('default_model','gpt-4.1')
                         """)

        await db.commit()


# ── Config helpers ─────────────────────────────────
async def get_config(key: str, default: str = "") -> str:
    async with aiosqlite.connect(get_db_path()) as db:
        async with db.execute(
                "SELECT value FROM config WHERE key = ?", (key,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else default


async def set_config(key: str, value: str):
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
            (key, value),
        )
        await db.commit()


async def delete_config(key: str):
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute("DELETE FROM config WHERE key = ?", (key,))
        await db.commit()


async def get_all_config() -> dict:
    async with aiosqlite.connect(get_db_path()) as db:
        async with db.execute("SELECT key, value FROM config") as cur:
            rows = await cur.fetchall()
    return {r[0]: r[1] for r in rows}


async def is_bootstrapped() -> bool:
    """Gateway coi là bootstrapped nếu có ít nhất 1 provider key."""
    cfg = await get_all_config()
    return any([
        cfg.get("openai_key"),
        cfg.get("gemini_key"),
        cfg.get("copilot_token"),
    ])


async def reset_config():
    """Xóa tất cả provider keys, giữ lại infra defaults."""
    INFRA_KEYS = {"chroma_host", "chroma_port", "ollama_host", "ollama_port", "embed_model", "default_model"}
    async with aiosqlite.connect(get_db_path()) as db:
        async with db.execute("SELECT key FROM config") as cur:
            rows = await cur.fetchall()
        for (key,) in rows:
            if key not in INFRA_KEYS:
                await db.execute("DELETE FROM config WHERE key = ?", (key,))
        await db.commit()


# ── Project helpers ────────────────────────────────
async def list_projects() -> list[dict]:
    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM projects ORDER BY created_at DESC") as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_project(name: str) -> dict | None:
    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
                "SELECT * FROM projects WHERE name = ?", (name,)
        ) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def add_project(name: str, path: str, description: str = "") -> dict:
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            "INSERT INTO projects (name, path, description) VALUES (?, ?, ?)",
            (name, path, description),
        )
        await db.commit()
    return await get_project(name)


async def update_project_indexed(name: str, file_count: int):
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            "UPDATE projects SET indexed = 1, file_count = ? WHERE name = ?",
            (file_count, name),
        )
        await db.commit()


async def delete_project(name: str):
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute("DELETE FROM projects WHERE name = ?", (name,))
        await db.commit()
