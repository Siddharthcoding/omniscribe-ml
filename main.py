import os
import asyncio
import logging
import traceback
import httpx
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from whisper_utils import transcribe_from_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ml-service] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="OmniScribe Transcription Service")

AUTH_TOKEN = os.environ.get("OMNISCRIBE_AUTH_TOKEN", "")


class TranscribeRequest(BaseModel):
    video_url: str
    video_id: str | None = None
    webhook_url: str | None = None


class TranscribeResponse(BaseModel):
    status: str


def verify_auth(authorization: str | None = Header(None)):
    if AUTH_TOKEN:
        if not authorization:
            raise HTTPException(401, "Missing Authorization header")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or token != AUTH_TOKEN:
            raise HTTPException(401, "Invalid authorization token")


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_endpoint(req: TranscribeRequest, authorization: str | None = Header(None)):
    verify_auth(authorization)
    logger.info("[transcribe] [video=%s] Accepted transcription request for %s, webhook=%s",
                req.video_id, req.video_url, req.webhook_url)
    asyncio.create_task(_process_and_webhook(req.video_url, req.video_id, req.webhook_url))
    return {"status": "accepted"}


async def _process_and_webhook(video_url: str, video_id: str | None, webhook_url: str | None):
    logger.info("[process] [video=%s] Starting transcription (will try YouTube API first, then Whisper)", video_id)
    try:
        result = await asyncio.to_thread(transcribe_from_url, video_url)
        seg_count = len(result.get("segments", []))
        source = result.get("source", "unknown")
        logger.info("[process] [video=%s] Transcription complete: %d segments, source=%s",
                    video_id, seg_count, source)
    except Exception as e:
        logger.error("[process] [video=%s] Transcription failed: %s\n%s",
                     video_id, e, traceback.format_exc())
        if video_id and webhook_url:
            logger.info("[process] [video=%s] Sending failure webhook to %s", video_id, webhook_url)
            await _send_webhook(webhook_url, {"success": False, "video_id": video_id, "error": str(e)})
        return

    payload = {**result, "video_id": video_id} if video_id else result
    if webhook_url:
        logger.info("[process] [video=%s] Sending success webhook to %s", video_id, webhook_url)
        await _send_webhook(webhook_url, payload)
        logger.info("[process] [video=%s] Webhook sent successfully", video_id)
    else:
        logger.info("[process] [video=%s] No webhook URL, result would be: %d segments", video_id, len(result.get("segments", [])))


async def _send_webhook(url: str, payload: dict):
    async with httpx.AsyncClient() as client:
        try:
            headers = {"Content-Type": "application/json"}
            supabase_anon_key = os.environ.get("SUPABASE_ANON_KEY")
            if supabase_anon_key:
                headers["Authorization"] = f"Bearer {supabase_anon_key}"
                logger.info("[webhook] Adding Authorization header (anon key present)")
            else:
                logger.warning("[webhook] No SUPABASE_ANON_KEY set, webhook call may be rejected by Supabase")
            resp = await client.post(url, json=payload, headers=headers, timeout=60)
            logger.info("[webhook] Sent to %s, status=%d", url, resp.status_code)
            if not resp.ok:
                body = await resp.aread()
                logger.warning("[webhook] Non-200 response: %s", body.decode()[:500])
        except Exception as e:
            logger.error("[webhook] Failed to send webhook to %s: %s", url, e)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/debug-transcribe")
async def debug_transcribe(req: TranscribeRequest, authorization: str | None = Header(None)):
    verify_auth(authorization)
    logger.info("[debug-transcribe] Transcribing %s synchronously", req.video_url)
    try:
        result = await asyncio.to_thread(transcribe_from_url, req.video_url)
        return result
    except Exception as e:
        logger.error("[debug-transcribe] Failed: %s", e)
        raise HTTPException(500, f"Transcription failed: {str(e)}")
