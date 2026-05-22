import whisper
import tempfile
import os
import subprocess
import re
import logging
import httpx
from typing import List, Dict

logger = logging.getLogger(__name__)

_model = None


def get_model():
    global _model
    if _model is None:
        logger.info("Loading Whisper tiny model...")
        _model = whisper.load_model("tiny")
        logger.info("Whisper tiny model loaded")
    return _model


def _extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def _parse_transcript_xml(xml: str) -> List[Dict] | None:
    segments = []

    for match in re.finditer(r'<text start="([\d.]+)" dur="([\d.]+)">(.*?)</text>', xml, re.DOTALL):
        text = match.group(3).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'").replace("&quot;", '"')
        start = float(match.group(1))
        dur = float(match.group(2))
        segments.append({"text": text, "start": start, "end": start + dur})

    if not segments:
        for match in re.finditer(r'<p t="([\d.]+)" d="([\d.]+)"[^>]*>(.*?)</p>', xml, re.DOTALL):
            content = re.sub(r"<[^>]+>", "", match.group(3))
            text = content.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'").replace("&quot;", '"')
            t = float(match.group(1)) / 1000
            d = float(match.group(2)) / 1000
            segments.append({"text": text, "start": t, "end": t + d})

    return segments if segments else None


def extract_youtube_transcript(video_url: str) -> List[Dict] | None:
    video_id = _extract_video_id(video_url)
    if not video_id:
        logger.info("[extract] Not a YouTube URL, skipping YouTube transcript extraction")
        return None

    logger.info("[extract] [video=%s] Attempting InnerTube API transcript fetch", video_id)
    try:
        import requests as req
        resp = req.post(
            "https://www.youtube.com/youtubei/v1/player?prettyPrint=false",
            json={
                "context": {
                    "client": {
                        "clientName": "ANDROID",
                        "clientVersion": "20.10.38",
                        "hl": "en",
                    }
                },
                "videoId": video_id,
            },
            headers={
                "User-Agent": "com.google.android.youtube/20.10.38 (Linux; U; Android 14)",
                "Content-Type": "application/json",
            },
            timeout=20,
            verify=False,
        )
        logger.info("[extract] [video=%s] InnerTube API responded with %s", video_id, resp.status_code)
        if resp.status_code == 200:
            data = resp.json()
            caption_tracks = data.get("captions", {}).get("playerCaptionsTracklistRenderer", {}).get("captionTracks", [])
            if caption_tracks:
                logger.info("[extract] [video=%s] Found %d caption tracks via InnerTube", video_id, len(caption_tracks))
                return _fetch_transcript_from_tracks(caption_tracks)
            else:
                logger.info("[extract] [video=%s] InnerTube returned 200 but no caption tracks found", video_id)
        else:
            logger.warning("[extract] [video=%s] InnerTube returned %s", video_id, resp.status_code)
    except Exception as e:
        logger.warning("[extract] [video=%s] InnerTube (requests) error: %s", video_id, e)

    logger.info("[extract] [video=%s] Falling back to youtubetranscript.com", video_id)
    try:
        resp = httpx.get(
            f"https://youtubetranscript.com/?v={video_id}",
            timeout=15,
            verify=False,
        )
        logger.info("[extract] [video=%s] youtubetranscript.com responded with %s", video_id, resp.status_code)
        if resp.status_code == 200 and resp.text.strip().startswith("<?xml"):
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            segments = []
            for child in root:
                text = (child.text or "").strip()
                start = float(child.get("start", 0))
                dur = float(child.get("dur", 0))
                if text:
                    segments.append({"text": text, "start": start, "end": start + dur})
            logger.info("[extract] [video=%s] Got %d segments from youtubetranscript.com", video_id, len(segments))
            return segments
        else:
            logger.warning("[extract] [video=%s] youtubetranscript.com returned %s (body start: %s)",
                           video_id, resp.status_code, resp.text[:100])
    except Exception as e:
        logger.warning("[extract] [video=%s] youtubetranscript.com error: %s", video_id, e)

    logger.info("[extract] [video=%s] No YouTube transcript available via API", video_id)
    return None


def _fetch_transcript_from_tracks(caption_tracks: List[Dict]) -> List[Dict] | None:
    track = None
    for t in caption_tracks:
        if t.get("languageCode") == "en":
            track = t
            break
    if not track and caption_tracks:
        track = caption_tracks[0]
    if not track:
        logger.warning("[tracks] No suitable caption track found")
        return None

    logger.info("[tracks] Fetching transcript from track: lang=%s, baseUrl=%s", track.get("languageCode"), track.get("baseUrl", "")[:80])
    try:
        import requests as req
        resp = req.get(track["baseUrl"], timeout=15, verify=False)
        if resp.status_code != 200:
            logger.warning("[tracks] Track fetch returned %s", resp.status_code)
            return None
        segments = _parse_transcript_xml(resp.text)
        logger.info("[tracks] Parsed %d segments from track", len(segments) if segments else 0)
        return segments
    except Exception as e:
        logger.warning("[tracks] Transcript track error: %s", e)
        return None


def _extract_audio(video_url: str) -> str:
    logger.info("[audio] [url=%s] Extracting audio via yt-dlp...", video_url)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    yt_dlp_path = os.environ.get("YT_DLP_PATH", "yt-dlp")
    logger.info("[audio] Using yt-dlp path: %s", yt_dlp_path)

    result = subprocess.run(
        [
            yt_dlp_path,
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", tmp.name.replace(".wav", ".%(ext)s"),
            "--print", "filename",
            "--no-check-certificates",
            "--impersonate", "chrome",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "--add-header", "Accept:text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "--add-header", "Accept-Language:en-US,en;q=0.9",
            "--add-header", "Origin:https://www.youtube.com",
            "--extractor-args", "youtube:player_client=android,web;skip=webpage",
            "--retries", "5",
            "--throttled-rate", "100M",
            video_url,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        stderr_preview = result.stderr[:2000] if result.stderr else "(no stderr)"
        logger.error("[audio] yt-dlp failed (exit %d): %s", result.returncode, stderr_preview)
        raise RuntimeError(
            f"yt-dlp failed (exit {result.returncode}):\n"
            f"stderr: {stderr_preview}"
        )

    logger.info("[audio] yt-dlp stdout: %s", result.stdout[:500])

    actual_file = result.stdout.strip()
    if actual_file and os.path.exists(actual_file):
        logger.info("[audio] Extracted audio to %s (from --print filename)", actual_file)
        return actual_file

    base = tmp.name.replace(".wav", "")
    for ext in [".wav", ".mp3", ".m4a", ".webm"]:
        candidate = base + ext
        if os.path.exists(candidate):
            logger.info("[audio] Found audio file: %s", candidate)
            return candidate

    raise FileNotFoundError(f"Could not find downloaded audio file ({tmp.name})")


def transcribe(audio_path: str) -> List[Dict]:
    logger.info("[whisper] Transcribing audio file %s...", audio_path)
    model = get_model()
    result = model.transcribe(audio_path, word_timestamps=True)
    logger.info("[whisper] Transcription complete: %d segments, language=%s", len(result["segments"]), result.get("language", "?"))

    segments = []
    for seg in result["segments"]:
        segments.append({
            "text": seg["text"].strip(),
            "start": seg["start"],
            "end": seg["end"],
        })

    return segments


def transcribe_from_url(video_url: str) -> dict:
    logger.info("[transcribe] Starting transcription for %s", video_url)

    # First try: YouTube API
    logger.info("[transcribe] Phase 1/3: Trying YouTube API transcript...")
    segments = extract_youtube_transcript(video_url)
    if segments:
        full_text = " ".join(s["text"] for s in segments)
        logger.info("[transcribe] Phase 1/3 SUCCESS: %d segments from YouTube API", len(segments))
        return {
            "segments": segments,
            "full_text": full_text,
            "language": "en",
            "source": "youtube_api",
        }

    # Second try: Whisper transcription
    logger.info("[transcribe] Phase 2/3: YouTube API failed, attempting Whisper transcription...")
    audio_path = None
    try:
        logger.info("[transcribe] Phase 2a/3: Extracting audio...")
        audio_path = _extract_audio(video_url)
        logger.info("[transcribe] Phase 2b/3: Running Whisper on %s...", audio_path)
        segments = transcribe(audio_path)
        full_text = " ".join(s["text"] for s in segments)
        logger.info("[transcribe] Phase 2/3 SUCCESS: %d segments from Whisper", len(segments))
        return {
            "segments": segments,
            "full_text": full_text,
            "language": "en",
            "source": "whisper",
        }
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            logger.info("[transcribe] Cleaned up temp audio file %s", audio_path)
