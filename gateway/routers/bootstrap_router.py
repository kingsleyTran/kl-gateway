"""
Bootstrap router — lưu API keys vào SQLite, không dùng .env.
"""

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db import (
    get_config, set_config, get_all_config,
    is_bootstrapped, reset_config
)

router = APIRouter()

GITHUB_CLIENT_ID  = "Iv1.b507a08c87ecfe98"
GITHUB_DEVICE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL  = "https://github.com/login/oauth/access_token"


# ── Schemas ───────────────────────────────────────
class OpenAISetup(BaseModel):
    api_key: str

class GeminiSetup(BaseModel):
    api_key: str

class SaveConfig(BaseModel):
    providers: dict   # {openai: {api_key}, gemini: {api_key}, copilot: {token}}
    default_model: str = "gpt-4.1"


# ── Routes ────────────────────────────────────────
@router.get("/status")
async def bootstrap_status():
    cfg = await get_all_config()
    return {
        "bootstrapped": await is_bootstrapped(),
        "providers": {
            "openai":  bool(cfg.get("openai_key")),
            "gemini":  bool(cfg.get("gemini_key")),
            "copilot": bool(cfg.get("copilot_token")),
        },
        "default_model": cfg.get("default_model", "gpt-4.1"),
    }


@router.post("/reset")
async def reset():
    await reset_config()
    return {"status": "reset"}


# ── OpenAI ────────────────────────────────────────
@router.post("/openai/verify")
async def verify_openai(body: OpenAISetup):
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {body.api_key}"},
            timeout=10,
        )
    if resp.status_code != 200:
        raise HTTPException(400, detail=f"Invalid OpenAI key: HTTP {resp.status_code}")

    all_models = resp.json().get("data", [])
    BLOCKED = ("embed", "tts", "dall", "whisper", "realtime", "-audio", "search", "moderat")
    models = sorted(
        {m["id"] for m in all_models if not any(k in m["id"] for k in BLOCKED)},
        key=lambda m: (2 if "mini" in m else 3 if "instruct" in m or "preview" in m else 4 if "nano" in m else 1)
    )
    return {"valid": True, "message": f"OpenAI verified — {len(models)} models", "models": models}


# ── Gemini ────────────────────────────────────────
@router.post("/gemini/verify")
async def verify_gemini(body: GeminiSetup):
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={body.api_key}",
            timeout=10,
        )
    if resp.status_code != 200:
        raise HTTPException(400, detail=f"Invalid Gemini key: HTTP {resp.status_code}")

    all_models = resp.json().get("models", [])
    ALLOWED = ("gemini-2", "gemini-3", "gemini-1.5")
    BLOCKED  = ("embed", "vision", "aqa")
    models = sorted({
        m["name"].replace("models/", "")
        for m in all_models
        if any(m["name"].replace("models/","").startswith(p) for p in ALLOWED)
           and not any(k in m["name"] for k in BLOCKED)
    })
    return {"valid": True, "message": f"Gemini verified — {len(models)} models", "models": models}


# ── GitHub Copilot Device Flow ────────────────────
@router.post("/copilot/start")
async def copilot_device_start():
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GITHUB_DEVICE_URL,
            data={"client_id": GITHUB_CLIENT_ID, "scope": "read:user"},
            headers={"Accept": "application/json"}, timeout=10,
        )
    if resp.status_code != 200:
        raise HTTPException(502, detail="Failed to start GitHub device flow")
    d = resp.json()
    return {
        "device_code":      d["device_code"],
        "user_code":        d["user_code"],
        "verification_uri": d["verification_uri"],
        "expires_in":       d["expires_in"],
        "interval":         d.get("interval", 5),
    }


@router.post("/copilot/poll")
async def copilot_device_poll(body: dict):
    device_code = body.get("device_code")
    if not device_code:
        raise HTTPException(400, detail="device_code required")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id":   GITHUB_CLIENT_ID,
                "device_code": device_code,
                "grant_type":  "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"}, timeout=10,
        )
    data = resp.json()
    if "access_token" in data:
        return {"status": "success", "token": data["access_token"]}
    error = data.get("error", "unknown")
    if error in ("expired_token", "access_denied"):
        raise HTTPException(400, detail=f"Device flow failed: {error}")
    return {"status": "slow_down" if error == "slow_down" else "pending"}


# ── Save — ghi vào SQLite ─────────────────────────
@router.post("/save")
async def save_bootstrap(body: SaveConfig):
    if body.providers.get("openai", {}).get("api_key"):
        await set_config("openai_key", body.providers["openai"]["api_key"])

    if body.providers.get("gemini", {}).get("api_key"):
        await set_config("gemini_key", body.providers["gemini"]["api_key"])

    if body.providers.get("copilot", {}).get("token"):
        await set_config("copilot_token", body.providers["copilot"]["token"])

    await set_config("default_model", body.default_model)

    return {
        "status":        "saved",
        "bootstrapped":  await is_bootstrapped(),
        "default_model": body.default_model,
    }
