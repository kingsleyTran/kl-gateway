import json
import logging
import time
from pathlib import Path

import httpx
from fastapi import HTTPException

from db import get_config, set_config

COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
DEFAULT_COPILOT_API_BASE_URL = "https://api.individual.githubcopilot.com"

logger = logging.getLogger("kl_gateway.copilot")


def derive_copilot_api_base_url_from_token(token: str) -> str:
    trimmed = (token or "").strip()
    if not trimmed:
        return DEFAULT_COPILOT_API_BASE_URL
    import re
    match = re.search(r"(?:^|;)\s*proxy-ep=([^;\s]+)", trimmed, re.IGNORECASE)
    proxy_ep = (match.group(1).strip() if match else "")
    if not proxy_ep:
        return DEFAULT_COPILOT_API_BASE_URL
    host = proxy_ep.replace("https://", "").replace("http://", "")
    if host.startswith("proxy."):
        host = "api." + host[len("proxy."):]
    return f"https://{host}" if host else DEFAULT_COPILOT_API_BASE_URL


def _cache_path(config_dir: Path) -> Path:
    return config_dir / "github-copilot.token.json"


def _load_cached_token(config_dir: Path) -> dict | None:
    path = _cache_path(config_dir)
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_cached_token(config_dir: Path, payload: dict):
    path = _cache_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _delete_cached_token(config_dir: Path):
    path = _cache_path(config_dir)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _is_token_usable(payload: dict, now_ms: int | None = None) -> bool:
    now_ms = now_ms or int(time.time() * 1000)
    expires_at = int(payload.get("expiresAt") or 0)
    return expires_at - now_ms > 300_000


def _parse_expires_at(value) -> int:
    if isinstance(value, (int, float)):
        num = int(value)
    elif isinstance(value, str) and value.strip():
        num = int(value.strip())
    else:
        raise ValueError("Copilot token response missing expires_at")
    return num if num > 10_000_000_000 else num * 1000


async def resolve_copilot_api_token(*, config_dir: Path, github_token: str, force_refresh: bool = False) -> dict:
    github_token = (github_token or "").strip()
    if not github_token:
        logger.warning("Copilot resolve requested without stored GitHub token")
        raise HTTPException(500, detail="Copilot is not configured. Connect GitHub in bootstrap first.")

    logger.info("Resolving Copilot API token (force_refresh=%s)", force_refresh)

    if not force_refresh:
        cached = _load_cached_token(config_dir)
        if cached and isinstance(cached.get("token"), str) and _is_token_usable(cached):
            token = cached["token"].strip()
            base_url = derive_copilot_api_base_url_from_token(token)
            logger.info("Using cached Copilot API token (expiresAt=%s, base_url=%s)", cached.get("expiresAt"), base_url)
            return {
                "token": token,
                "expires_at": int(cached["expiresAt"]),
                "source": "cache",
                "base_url": base_url,
            }

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            COPILOT_TOKEN_URL,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {github_token}",
            },
            timeout=20,
        )

    logger.info("Copilot token exchange HTTP %s", resp.status_code)
    if resp.status_code == 401:
        logger.warning("Copilot token exchange returned 401")
        raise HTTPException(401, detail="GitHub token is invalid or expired. Reconnect GitHub Copilot in bootstrap first.")
    if resp.status_code >= 400:
        body_preview = resp.text[:500]
        logger.warning("Copilot token exchange failed: HTTP %s body=%s", resp.status_code, body_preview)
        raise HTTPException(502, detail=f"Copilot token exchange failed: HTTP {resp.status_code} — {body_preview}")

    data = resp.json()
    logger.info("Copilot token exchange keys=%s", sorted(list(data.keys())))
    token = str(data.get("token") or "").strip()
    if not token:
        logger.warning("Copilot token exchange returned no token payload=%s", data)
        raise HTTPException(502, detail="Copilot token exchange succeeded but returned no token.")

    try:
        expires_at = _parse_expires_at(data.get("expires_at"))
    except Exception as exc:
        logger.warning("Copilot token exchange returned invalid expiry payload=%s error=%s", data, exc)
        raise HTTPException(502, detail=f"Copilot token exchange returned invalid expiry: {exc}")

    base_url = derive_copilot_api_base_url_from_token(token)
    payload = {
        "token": token,
        "expiresAt": expires_at,
        "updatedAt": int(time.time() * 1000),
    }
    _save_cached_token(config_dir, payload)
    await set_config("copilot_api_base_url", base_url)
    logger.info("Resolved Copilot API token successfully (base_url=%s expires_at=%s)", base_url, expires_at)

    return {
        "token": token,
        "expires_at": expires_at,
        "source": "fetched",
        "base_url": base_url,
    }


async def get_resolved_copilot_credentials(*, config_dir: Path, force_refresh: bool = False) -> dict:
    github_token = await get_config("copilot_token")
    return await resolve_copilot_api_token(config_dir=config_dir, github_token=github_token, force_refresh=force_refresh)


async def invalidate_copilot_api_token_cache(*, config_dir: Path):
    _delete_cached_token(config_dir)
    await set_config("copilot_api_base_url", "")
