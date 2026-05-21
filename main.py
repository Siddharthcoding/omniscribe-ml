import os
import asyncio
import httpx
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from whisper_utils import transcribe_from_url

app = FastAPI(title="OmniScribe Transcription Service")

AUTH_TOKEN = os.environ.get("OMNISCRIBE_AUTH_TOKEN", "")


class TranscribeRequest(BaseModel):
    video_url: str
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
    asyncio.create_task(_process_and_webhook(req.video_url, req.webhook_url))
    return {"status": "accepted"}


async def _process_and_webhook(video_url: str, webhook_url: str | None):
    try:
        result = await asyncio.to_thread(transcribe_from_url, video_url)
    except Exception as e:
        print(f"Transcription failed: {e}")
        return

    if webhook_url:
        await _send_webhook(webhook_url, result)


async def _send_webhook(url: str, payload: dict):
    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, json=payload, timeout=60)
        except Exception as e:
            print(f"Webhook failed: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}
