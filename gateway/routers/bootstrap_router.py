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


# ── OpenAI Codex OAuth (ChatGPT subscription) ─────
import secrets as _secrets
import hashlib as _hashlib
import base64 as _base64

# In-memory store for ongoing OAuth flows
_oauth_states: dict = {}

OPENAI_CLIENT_ID  = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_AUTH_URL   = "https://auth.openai.com/oauth/authorize"
OPENAI_TOKEN_URL  = "https://auth.openai.com/oauth/token"
OPENAI_REDIRECT   = "http://127.0.0.1:1455/auth/callback"


@router.get("/openai/oauth/start")
async def openai_oauth_start():
    """
    Generate PKCE params + spin up temp HTTP server on port 1455
    to catch the OAuth callback from OpenAI.
    """
    import asyncio
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlencode, urlparse, parse_qs

    code_verifier  = _secrets.token_urlsafe(64)
    digest         = _hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = _base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    state          = _secrets.token_hex(16)

    _oauth_states[state] = {"verifier": code_verifier, "done": False, "error": None}
    flow = _oauth_states[state]

    # Spin up temp server on 1455 in background thread
    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != "/auth/callback":
                self.send_response(404); self.end_headers(); return

            params   = parse_qs(parsed.query)
            cb_code  = params.get("code", [None])[0]
            cb_state = params.get("state", [None])[0]
            cb_error = params.get("error", [None])[0]

            if cb_error:
                flow["error"] = cb_error
                flow["done"]  = True
                self.send_response(200); self.send_header("Content-Type","text/html"); self.end_headers()
                self.wfile.write(b"<html><body><h2>Auth error</h2><script>window.close()</script></body></html>")
                return

            if cb_state != state:
                flow["error"] = "state_mismatch"
                flow["done"]  = True
                self.send_response(400); self.end_headers(); return

            # Exchange code for token (sync inside handler thread)
            import urllib.request, json as _j
            payload = _j.dumps({
                "grant_type":    "authorization_code",
                "client_id":     OPENAI_CLIENT_ID,
                "code":          cb_code,
                "redirect_uri":  OPENAI_REDIRECT,
                "code_verifier": flow["verifier"],
            }).encode()
            req = urllib.request.Request(
                OPENAI_TOKEN_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as r:
                    token_data = _j.loads(r.read())
                flow["token_data"] = token_data
            except Exception as e:
                flow["error"] = str(e)

            flow["done"] = True
            self.send_response(200); self.send_header("Content-Type","text/html"); self.end_headers()
            self.wfile.write(b"""<html><body style="font-family:monospace;background:#0a0a0f;color:#00ff9d;padding:40px;text-align:center">
                <h2>Connected!</h2><p>You can close this tab.</p>
                <script>window.opener?.postMessage({type:'openai_codex_auth_done'},'*');setTimeout(()=>window.close(),1000)</script>
            </body></html>""")

        def log_message(self, *args): pass  # suppress logs

    def run_server():
        try:
            srv = HTTPServer(("127.0.0.1", 1455), CallbackHandler)
            srv.timeout = 300  # 5 min timeout
            srv.handle_request()  # handle exactly 1 request then stop
        except Exception:
            pass

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    params = {
        "response_type":         "code",
        "client_id":             OPENAI_CLIENT_ID,
        "redirect_uri":          OPENAI_REDIRECT,
        "scope":                 "openid profile email offline_access model.request api.model.read",
        "state":                 state,
        "code_challenge":        code_challenge,
        "code_challenge_method": "S256",
    }
    url = OPENAI_AUTH_URL + "?" + urlencode(params)
    return {"url": url, "state": state}


@router.get("/openai/oauth/poll")
async def openai_oauth_poll(state: str):
    """Poll whether OAuth flow completed."""
    import time as _time
    flow = _oauth_states.get(state)
    if not flow:
        return {"status": "not_found"}
    if flow.get("error"):
        _oauth_states.pop(state, None)
        return {"status": "error", "error": flow["error"]}
    if flow.get("done") and flow.get("token_data"):
        td = flow.pop("token_data")
        _oauth_states.pop(state, None)
        access  = td.get("access_token", "")
        refresh = td.get("refresh_token", "")
        exp     = td.get("expires_in", 3600)
        await set_config("openai_codex_access",  access)
        await set_config("openai_codex_refresh", refresh)
        await set_config("openai_codex_expires", str(int(_time.time()) + exp))
        return {"status": "success"}
    return {"status": "pending"}


@router.get("/openai/oauth/callback")
async def openai_oauth_callback(code: str = "", state: str = "", error: str = ""):
    """Fallback callback trên gateway port (backup)."""
    from fastapi.responses import HTMLResponse
    if error:
        return HTMLResponse(f"<html><body>Error: {error}</body></html>")
    return HTMLResponse("<html><body>Please use the primary callback on port 1455.</body></html>")


@router.get("/openai/oauth/status")
async def openai_oauth_status():
    """Check if Codex OAuth token is valid."""
    import time as _time
    access  = await get_config("openai_codex_access",  "")
    expires = await get_config("openai_codex_expires", "0")
    valid   = bool(access) and int(expires) > int(_time.time())
    return {"valid": valid, "has_token": bool(access)}


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


# GitHub Copilot OAuth bị GitHub block (403 Forbidden).
# Removed — dùng OpenAI hoặc Gemini thay thế.


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


