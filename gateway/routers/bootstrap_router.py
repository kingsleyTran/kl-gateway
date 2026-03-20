"""
Bootstrap router — lưu API keys / OAuth tokens vào SQLite, không dùng .env.
"""

import os
import secrets

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from db import (
    get_config, set_config, get_all_config,
    is_bootstrapped, reset_config, save_oauth_state, pop_oauth_state
)
from oauth import (
    OPENAI_CODEX_API_BASE,
    OPENAI_CODEX_REDIRECT_URI,
    build_openai_codex_authorize_url,
    exchange_openai_codex_code,
    extract_code_from_callback_url,
    generate_pkce_pair,
)

router = APIRouter()

GITHUB_CLIENT_ID  = "Iv1.b507a08c87ecfe98"
GITHUB_DEVICE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL  = "https://github.com/login/oauth/access_token"


# ── Schemas ───────────────────────────────────────
class OpenAISetup(BaseModel):
    api_key: str
    test_model: str | None = None

class GeminiSetup(BaseModel):
    api_key: str

class OpenAICodexCallback(BaseModel):
    callback_url: str

class SaveConfig(BaseModel):
    providers: dict   # {openai: {api_key}, openai_codex: {...}, gemini: {api_key}, copilot: {token}}
    default_model: str = "openai/gpt-4.1"


# ── Routes ────────────────────────────────────────
@router.get("/status")
async def bootstrap_status():
    cfg = await get_all_config()
    return {
        "bootstrapped": await is_bootstrapped(),
        "providers": {
            "openai": bool(cfg.get("openai_key")),
            "openai_codex": bool(cfg.get("openai_codex_refresh_token")),
            "gemini": bool(cfg.get("gemini_key")),
            "copilot": bool(cfg.get("copilot_token")),
        },
        "default_model": cfg.get("default_model", "openai/gpt-4.1"),
    }


@router.post("/reset")
async def reset():
    await reset_config()
    return {"status": "reset"}


# ── OpenAI ────────────────────────────────────────
@router.post("/openai/verify")
async def verify_openai(body: OpenAISetup):
    headers = {"Authorization": f"Bearer {body.api_key}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.openai.com/v1/models",
            headers=headers,
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

    preferred = [
        body.test_model,
        "gpt-4.1-mini",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4o",
        "gpt-5.4",
        "gpt-5.3-codex",
        "codex-mini-latest",
    ]
    runtime_model = next((m for m in preferred if m and m in models), None)
    if not runtime_model:
        raise HTTPException(400, detail="OpenAI key is valid, but no supported chat model was found for runtime use.")

    async with httpx.AsyncClient() as client:
        test_resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "model": runtime_model,
                "messages": [{"role": "user", "content": "Reply with OK"}],
                "max_tokens": 5,
            },
            timeout=20,
        )

    if test_resp.status_code == 429:
        detail = test_resp.json().get("error", {}).get("message", "OpenAI quota exceeded")
        raise HTTPException(400, detail=f"OpenAI key listed models, but runtime test failed with 429: {detail}")
    if test_resp.status_code >= 400:
        detail = test_resp.json().get("error", {}).get("message", f"HTTP {test_resp.status_code}")
        raise HTTPException(400, detail=f"OpenAI key listed models, but runtime test on {runtime_model} failed: {detail}")

    return {
        "valid": True,
        "message": f"OpenAI verified — {len(models)} models, runtime OK on {runtime_model}",
        "models": [f"openai/{m}" for m in models],
        "runtime_model": f"openai/{runtime_model}",
    }


# ── Gemini ────────────────────────────────────────
@router.post("/openai-codex/start")
async def start_openai_codex_oauth(request: Request):
    state = secrets.token_urlsafe(24)
    verifier, challenge = generate_pkce_pair()
    authorize_url = build_openai_codex_authorize_url(state=state, code_challenge=challenge)
    await save_oauth_state("openai-codex", state, {
        "code_verifier": verifier,
        "redirect_uri": OPENAI_CODEX_REDIRECT_URI,
        "created_from": str(request.base_url),
    })
    return {
        "authorize_url": authorize_url,
        "state": state,
        "redirect_uri": OPENAI_CODEX_REDIRECT_URI,
        "instructions": "Sign in with ChatGPT/Codex, then paste the final callback URL here if the browser does not return automatically.",
    }


async def _store_openai_codex_tokens(token_data: dict):
    await set_config("openai_codex_access_token", token_data["access_token"])
    await set_config("openai_codex_refresh_token", token_data["refresh_token"])
    await set_config("openai_codex_expires_at", str(token_data["expires_at"]))
    await set_config("openai_codex_account_id", token_data.get("account_id", ""))
    await set_config("openai_codex_token_type", token_data.get("token_type", "Bearer"))
    await set_config("openai_codex_scope", token_data.get("scope", ""))


async def _exchange_openai_codex_callback(callback_url: str) -> dict:
    params = extract_code_from_callback_url(callback_url)
    if params.get("error"):
        detail = params.get("error_description") or params["error"]
        raise HTTPException(400, detail=f"OpenAI OAuth was declined: {detail}")
    code = params.get("code")
    state = params.get("state")
    if not code or not state:
        raise HTTPException(400, detail="Callback URL must include both code and state.")

    saved = await pop_oauth_state("openai-codex", state)
    if not saved:
        raise HTTPException(400, detail="OAuth session expired or is invalid. Start the login flow again.")

    try:
        token_data = await exchange_openai_codex_code(
            code=code,
            code_verifier=saved["code_verifier"],
            redirect_uri=saved.get("redirect_uri") or OPENAI_CODEX_REDIRECT_URI,
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response is not None else str(exc)
        raise HTTPException(400, detail=f"OpenAI token exchange failed. {detail}")
    except httpx.HTTPError as exc:
        raise HTTPException(502, detail=f"OpenAI OAuth network error: {exc}")

    await _store_openai_codex_tokens(token_data)

    return {
        "valid": True,
        "message": "OpenAI Codex OAuth connected successfully.",
        "account_id": token_data.get("account_id", ""),
        "expires_at": token_data["expires_at"],
        "api_base": OPENAI_CODEX_API_BASE,
    }


@router.post("/openai-codex/exchange")
async def exchange_openai_codex_callback(body: OpenAICodexCallback):
    return await _exchange_openai_codex_callback(body.callback_url)


@router.get("/auth/callback", response_class=HTMLResponse)
async def openai_codex_callback(code: str = "", state: str = "", error: str = "", error_description: str = ""):
    callback_url = f"{OPENAI_CODEX_REDIRECT_URI}?code={code}&state={state}&error={error}&error_description={error_description}"
    try:
        result = await _exchange_openai_codex_callback(callback_url)
        return f"<html><body style='font-family: sans-serif; padding: 24px;'><h2>OpenAI Codex connected</h2><p>{result['message']}</p><p>You can close this tab and return to the gateway setup.</p></body></html>"
    except HTTPException as exc:
        return HTMLResponse(
            f"<html><body style='font-family: sans-serif; padding: 24px;'><h2>OAuth failed</h2><p>{exc.detail}</p><p>Return to the gateway setup page and try again.</p></body></html>",
            status_code=exc.status_code,
        )


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
    return {"valid": True, "message": f"Gemini verified — {len(models)} models", "models": [f"gemini/{m}" for m in models]}


# ── GitHub Copilot Device Flow ────────────────────
async def ensure_copilot_backend() -> str:
    copilot_url = os.getenv("COPILOT_API_URL", "http://localhost:4000")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{copilot_url}/v1/models", timeout=5)
        except httpx.HTTPError:
            raise HTTPException(
                503,
                detail=(
                    f"Copilot backend is not running at {copilot_url}. "
                    f"This gateway expects an OpenAI-compatible Copilot bridge exposing /v1/models there."
                ),
            )
    if resp.status_code != 200:
        raise HTTPException(
            503,
            detail=(
                f"Copilot backend at {copilot_url} responded with HTTP {resp.status_code}. "
                f"Expected an OpenAI-compatible bridge."
            ),
        )
    return copilot_url


@router.get("/copilot/status")
async def copilot_status():
    try:
        url = await ensure_copilot_backend()
        return {"available": True, "url": url}
    except HTTPException as e:
        return {"available": False, "detail": e.detail}


@router.post("/copilot/start")
async def copilot_device_start():
    await ensure_copilot_backend()
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
    await ensure_copilot_backend()
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

    if body.providers.get("openai_codex", {}).get("access_token") not in (None, "", "stored-via-exchange"):
        await set_config("openai_codex_access_token", body.providers["openai_codex"]["access_token"])
    if body.providers.get("openai_codex", {}).get("refresh_token") not in (None, "", "stored-via-exchange"):
        await set_config("openai_codex_refresh_token", body.providers["openai_codex"]["refresh_token"])
    if body.providers.get("openai_codex", {}).get("expires_at"):
        await set_config("openai_codex_expires_at", str(body.providers["openai_codex"]["expires_at"]))
    if body.providers.get("openai_codex", {}).get("account_id") is not None:
        await set_config("openai_codex_account_id", body.providers["openai_codex"].get("account_id", ""))

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
