import base64
import hashlib
import secrets
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

OPENAI_CODEX_CLIENT_ID = "openai-codex"
OPENAI_CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_CODEX_REDIRECT_URI = "http://127.0.0.1:1455/auth/callback"
OPENAI_CODEX_SCOPES = "openid offline_access profile email"
OPENAI_CODEX_API_BASE = "https://api.openai.com/v1"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def generate_pkce_pair() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(48))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge


def build_openai_codex_authorize_url(*, state: str, code_challenge: str, redirect_uri: str = OPENAI_CODEX_REDIRECT_URI) -> str:
    from urllib.parse import urlencode

    params = {
        "response_type": "code",
        "client_id": OPENAI_CODEX_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": OPENAI_CODEX_SCOPES,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return f"{OPENAI_CODEX_AUTHORIZE_URL}?{urlencode(params)}"


def extract_code_from_callback_url(callback_url: str) -> dict[str, str]:
    parsed = urlparse(callback_url.strip())
    query = parse_qs(parsed.query)
    code = (query.get("code") or [""])[0]
    state = (query.get("state") or [""])[0]
    error = (query.get("error") or [""])[0]
    error_description = (query.get("error_description") or [""])[0]
    return {
        "code": code,
        "state": state,
        "error": error,
        "error_description": error_description,
    }


def decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        data = base64.urlsafe_b64decode(payload.encode())
        import json
        return json.loads(data.decode())
    except Exception:
        return {}


def _normalize_token_payload(data: dict[str, Any]) -> dict[str, Any]:
    access_token = str(data.get("access_token") or "").strip()
    refresh_token = str(data.get("refresh_token") or "").strip()
    expires_in = int(data.get("expires_in") or 3600)
    id_token = str(data.get("id_token") or "").strip()

    account_id = ""
    payload = decode_jwt_payload(access_token) if access_token else {}
    for key in ("account_id", "accountId", "sub", "org_id", "orgId"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            account_id = val.strip()
            break
    if not account_id and id_token:
        payload = decode_jwt_payload(id_token)
        for key in ("account_id", "accountId", "sub", "org_id", "orgId"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                account_id = val.strip()
                break

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": int(time.time()) + max(expires_in - 30, 0),
        "expires_in": expires_in,
        "account_id": account_id,
        "token_type": str(data.get("token_type") or "Bearer"),
        "scope": str(data.get("scope") or OPENAI_CODEX_SCOPES),
        "id_token": id_token,
    }


async def exchange_openai_codex_code(*, code: str, code_verifier: str, redirect_uri: str = OPENAI_CODEX_REDIRECT_URI) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            OPENAI_CODEX_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": OPENAI_CODEX_CLIENT_ID,
                "code": code,
                "code_verifier": code_verifier,
                "redirect_uri": redirect_uri,
            },
            headers={"Accept": "application/json"},
            timeout=20,
        )
    resp.raise_for_status()
    return _normalize_token_payload(resp.json())


async def refresh_openai_codex_token(*, refresh_token: str) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            OPENAI_CODEX_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": OPENAI_CODEX_CLIENT_ID,
                "refresh_token": refresh_token,
            },
            headers={"Accept": "application/json"},
            timeout=20,
        )
    resp.raise_for_status()
    return _normalize_token_payload(resp.json())
