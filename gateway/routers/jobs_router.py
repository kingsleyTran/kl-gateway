"""
Job queue router — enqueue, cancel, status via arq + Redis.
SSE endpoint stream progress từ Redis pub/sub.
"""

import os
import json
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from arq import create_pool
from arq.connections import RedisSettings
import redis.asyncio as aioredis

router = APIRouter()

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")


async def get_redis():
    return await aioredis.from_url(REDIS_URL, decode_responses=True)


async def get_arq():
    return await create_pool(RedisSettings.from_dsn(REDIS_URL))


# ── Enqueue ───────────────────────────────────────
@router.post("/jobs/index")
async def enqueue_index(body: dict):
    """Enqueue index job, return job_id ngay lập tức."""
    project    = body.get("project")
    extensions = body.get("extensions", ".py,.ts,.js,.go,.java,.rs,.tsx,.jsx")

    if not project:
        raise HTTPException(400, detail="project required")

    pool   = await get_arq()
    job    = await pool.enqueue_job("index_project", project, extensions)
    await pool.aclose()

    return {"job_id": job.job_id, "project": project, "status": "queued"}


# ── Cancel ────────────────────────────────────────
@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Set cancel flag trong Redis — worker sẽ stop sau batch hiện tại."""
    r = await get_redis()
    await r.set("index:cancel:" + job_id, "1", ex=300)
    await r.aclose()
    return {"job_id": job_id, "status": "cancel_requested"}


# ── Status ────────────────────────────────────────
@router.get("/jobs/{job_id}/status")
async def job_status(job_id: str):
    """Lấy status + result của job."""
    pool = await get_arq()
    job  = await pool.job(job_id)
    if not job:
        raise HTTPException(404, detail="Job not found")

    info = await job.info()
    result = None
    try:
        result = await job.result(timeout=0)
    except Exception:
        pass

    await pool.aclose()
    return {
        "job_id":   job_id,
        "status":   info.status if info else "unknown",
        "result":   result,
        "enqueue_time": str(info.enqueue_time) if info else None,
        "start_time":   str(info.start_time)   if info and info.start_time else None,
    }


# ── List jobs ────────────────────────────────────
@router.get("/jobs")
async def list_jobs():
    """List tất cả jobs đang queued/running."""
    r = await get_redis()
    keys = await r.keys("arq:job:*")
    jobs = []
    for key in keys[:50]:
        try:
            data = await r.hgetall(key)
            jobs.append({
                "job_id":  key.replace("arq:job:", ""),
                "status":  data.get("status", "unknown"),
                "function": data.get("function", ""),
                "args":    data.get("args", ""),
            })
        except Exception:
            pass
    await r.aclose()
    return {"jobs": jobs}


# ── SSE progress stream ───────────────────────────
@router.get("/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    """Stream progress events từ Redis pub/sub."""
    SEP = chr(10) + chr(10)

    async def event_stream():
        r = await get_redis()
        channel = "index:progress:" + job_id

        try:
            pubsub = r.pubsub()
            await pubsub.subscribe(channel)

            # Send queued event ngay
            yield "data: " + json.dumps({"type": "subscribed", "job_id": job_id}) + SEP

            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue

                data = json.loads(message["data"])
                yield "data: " + json.dumps(data) + SEP

                if data.get("type") in ("done", "cancelled", "error"):
                    break

        finally:
            await pubsub.unsubscribe(channel)
            await r.aclose()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
