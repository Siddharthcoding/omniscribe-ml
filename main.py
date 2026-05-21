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
    segments: list[dict]
    full_text: str
    language: str


def verify_auth(authorization: str | None = Header(None)):
    if AUTH_TOKEN:
        if not authorization:
            raise HTTPException(401, "Missing Authorization header")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or token != AUTH_TOKEN:
            raise HTTPException(401, "Invalid authorization token")


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_endpoint(req: TranscribeRequest, auth=Header(None)):
    verify_auth(auth)

    try:
        result = await asyncio.to_thread(transcribe_from_url, req.video_url)
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")

    if req.webhook_url:
        asyncio.create_task(_send_webhook(req.webhook_url, result))

    return result


async def _send_webhook(url: str, payload: dict):
    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, json=payload, timeout=30)
        except Exception:
            pass  # fire-and-forget


@app.get("/health")
async def health():
    return {"status": "ok"}
