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
    # Use new google-genai SDK to list models
    try:
        from google import genai as google_genai
        client = google_genai.Client(api_key=body.api_key)
        all_models = list(client.models.list())
        BLOCKED = ("embed", "vision", "aqa", "text-", "chat-")
        models = sorted({
            m.name.replace("models/", "")
            for m in all_models
            if not any(k in m.name for k in BLOCKED)
               and "gemini" in m.name
        })
        return {"valid": True, "message": f"Gemini verified — {len(models)} models", "models": models}
    except Exception as e:
        raise HTTPException(400, detail=f"Invalid Gemini key: {str(e)}")


# GitHub Copilot OAuth bị GitHub block (403 Forbidden).
# Removed — dùng OpenAI hoặc Gemini thay thế.


# ── Copilot Auth Flow ─────────────────────────────
# copilot-api expose endpoints:
#   POST /api/auth/initiate  → { userCode, verificationUrl, deviceCode, interval }
#   POST /api/auth/poll      → { status: "pending"|"success", token? }

@router.get("/copilot/status")
async def copilot_status():
    """Check copilot-api sidecar status + auth state."""
    import os
    copilot_url = os.getenv("COPILOT_API_URL", "http://copilot-api:4141")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{copilot_url}/v1/models", timeout=5)
        if resp.status_code == 200:
            models = [m["id"] for m in resp.json().get("data", [])]
            return {"online": True, "authed": True, "models": models}
        return {"online": True, "authed": False}
    except httpx.ConnectError:
        return {"online": False, "authed": False}
    except Exception as e:
        return {"online": False, "authed": False, "error": str(e)}


@router.post("/copilot/auth/start")
async def copilot_auth_start():
    """
    Trigger GitHub device flow qua copilot-api sidecar.
    copilot-api dùng GitHub device flow natively —
    gateway chạy device flow trực tiếp thay vì relay qua sidecar.
    """
    # Dùng GitHub device flow trực tiếp (giống flow cũ)
    # copilot-api sẽ tự pick up token từ volume sau khi auth
    GITHUB_CLIENT_ID  = "Iv1.b507a08c87ecfe98"
    GITHUB_DEVICE_URL = "https://github.com/login/device/code"

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GITHUB_DEVICE_URL,
            data={"client_id": GITHUB_CLIENT_ID, "scope": "read:user"},
            headers={"Accept": "application/json"},
            timeout=10,
        )
    if resp.status_code != 200:
        raise HTTPException(502, detail="GitHub device flow failed")
    d = resp.json()
    return {
        "user_code":        d["user_code"],
        "verification_url": d["verification_uri"],
        "device_code":      d["device_code"],
        "interval":         d.get("interval", 5),
        "expires_in":       d.get("expires_in", 899),
    }


@router.post("/copilot/auth/poll")
async def copilot_auth_poll(body: dict):
    """
    Poll GitHub token.
    Khi có token → lưu vào DB + ghi ra .env để copilot-api container pick up.
    """
    import os, subprocess
    device_code = body.get("device_code", "")
    GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"
    GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id":   GITHUB_CLIENT_ID,
                "device_code": device_code,
                "grant_type":  "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
            timeout=10,
        )
    data = resp.json()

    if "access_token" in data:
        token = data["access_token"]

        # 1. Lưu vào SQLite
        await set_config("copilot_enabled", "1")
        await set_config("copilot_github_token", token)

        # 2. Ghi token vào 2 file:
        #    - github_token : copilot-api đọc trực tiếp
        #    - env          : copilot-api đọc qua env_file khi restart
        import os as _os
        import asyncio as _aio

        token_dir = "/copilot-data"
        try:
            _os.makedirs(token_dir, exist_ok=True)

            # File token trực tiếp
            token_path = f"{token_dir}/github_token"
            with open(token_path, "w") as f:
                f.write(token)
            _os.chmod(token_path, 0o600)

            # File env cho docker env_file — persist qua restart
            env_path = f"{token_dir}/env"
            with open(env_path, "w") as f:
                f.write("GITHUB_TOKEN=" + token + chr(10))
            _os.chmod(env_path, 0o600)

        except Exception as e:
            print(f"Warning: could not write token files: {e}")

        # 3. Restart copilot-api để pick up token
        try:
            proc = await _aio.create_subprocess_exec(
                "docker", "restart", "copilot-api",
                stdout=_aio.subprocess.PIPE,
                stderr=_aio.subprocess.PIPE,
            )
            await proc.communicate()
            return {"status": "success", "restarted": True}
        except Exception:
            return {"status": "success", "restarted": False,
                    "note": "Run manually: docker restart copilot-api"}

    error = data.get("error", "unknown")
    if error in ("expired_token", "access_denied"):
        raise HTTPException(400, detail=f"Auth failed: {error}")
    return {"status": "slow_down" if error == "slow_down" else "pending"}


@router.post("/copilot/restart")
async def copilot_restart():
    """
    Restart copilot-api để apply token mới.
    Dùng docker socket nếu available, fallback về hướng dẫn thủ công.
    """
    import asyncio as _asyncio
    try:
        proc = await _asyncio.create_subprocess_exec(
            "docker", "restart", "copilot-api",
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return {"status": "restarted"}
    except FileNotFoundError:
        # docker CLI không có trong container — user restart thủ công
        return {
            "status": "manual",
            "message": "Run: docker restart copilot-api"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/copilot/verify")
async def verify_copilot():
    """Verify copilot-api accessible và đã auth."""
    import os
    copilot_url = os.getenv("COPILOT_API_URL", "http://copilot-api:4141")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{copilot_url}/v1/models", timeout=5)
        if resp.status_code != 200:
            raise HTTPException(400, detail=f"copilot-api HTTP {resp.status_code} — need to auth first")
        models = [m["id"] for m in resp.json().get("data", [])]
        await set_config("copilot_enabled", "1")
        return {"valid": True, "message": f"Copilot ready — {len(models)} models", "models": models}
    except httpx.ConnectError:
        raise HTTPException(503, detail="copilot-api not reachable")
    except Exception as e:
        raise HTTPException(503, detail=str(e))


# ── Save — ghi vào SQLite ─────────────────────────
@router.post("/save")
async def save_bootstrap(body: SaveConfig):
    if body.providers.get("openai", {}).get("api_key"):
        await set_config("openai_key", body.providers["openai"]["api_key"])

    if body.providers.get("gemini", {}).get("api_key"):
        await set_config("gemini_key", body.providers["gemini"]["api_key"])

    if body.providers.get("copilot", {}).get("enabled"):
        await set_config("copilot_enabled", "1")

    if body.providers.get("copilot", {}).get("token"):
        await set_config("copilot_token", body.providers["copilot"]["token"])

    await set_config("default_model", body.default_model)

    return {
        "status":        "saved",
        "bootstrapped":  await is_bootstrapped(),
        "default_model": body.default_model,
    }
