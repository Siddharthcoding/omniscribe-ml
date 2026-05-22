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


def _try_innertube_client(video_id: str, client_name: str, client_version: str) -> List[Dict] | None:
    """Try to fetch captions via InnerTube API with a specific client."""
    import requests as req
    user_agent = f"Mozilla/5.0 (compatible; {client_name}/{client_version})"
    if client_name == "ANDROID":
        user_agent = f"com.google.android.youtube/{client_version} (Linux; U; Android 14)"

    resp = req.post(
        "https://www.youtube.com/youtubei/v1/player?prettyPrint=false",
        json={
            "context": {
                "client": {
                    "clientName": client_name,
                    "clientVersion": client_version,
                    "hl": "en",
                }
            },
            "videoId": video_id,
        },
        headers={
            "User-Agent": user_agent,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        timeout=20,
        verify=False,
    )
    if resp.status_code != 200:
        logger.info("[innertube] [video=%s] client=%s responded with %s", video_id, client_name, resp.status_code)
        return None

    data = resp.json()
    # Log response structure for debugging
    top_keys = list(data.keys())
    logger.info("[innertube] [video=%s] client=%s response keys: %s", video_id, client_name, top_keys)

    # Try multiple paths to find caption tracks
    captions = data.get("captions") or {}
    if isinstance(captions, dict):
        tracklist = captions.get("playerCaptionsTracklistRenderer") or {}
        caption_tracks = tracklist.get("captionTracks") or []
        if caption_tracks:
            logger.info("[innertube] [video=%s] client=%s found %d tracks via captions path", video_id, client_name, len(caption_tracks))
            return _fetch_transcript_from_tracks(caption_tracks)
        else:
            logger.info("[innertube] [video=%s] client=%s captionTracks empty, tracklist keys: %s", video_id, client_name, list(tracklist.keys()))
    else:
        logger.info("[innertube] [video=%s] client=%s 'captions' key missing or not a dict", video_id, client_name)

    return None


def extract_youtube_transcript(video_url: str) -> List[Dict] | None:
    video_id = _extract_video_id(video_url)
    if not video_id:
        logger.info("[extract] Not a YouTube URL, skipping YouTube transcript extraction")
        return None

    # Try multiple InnerTube clients
    clients = [
        ("ANDROID", "20.10.38"),
        ("WEB", "2.20231201"),
        ("WEB_EMBEDDED_PLAYER", "1.0"),
        ("ANDROID_MUSIC", "6.42.52"),
    ]
    for client_name, client_version in clients:
        logger.info("[extract] [video=%s] Trying InnerTube client=%s/%s", video_id, client_name, client_version)
        segments = _try_innertube_client(video_id, client_name, client_version)
        if segments:
            return segments

    # Fallback: youtube-transcript-api library (might work on Render's IP)
    logger.info("[extract] [video=%s] InnerTube failed, trying youtube-transcript-api library", video_id)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api.formatters import JSONFormatter
        fetched = YouTubeTranscriptApi.get_transcript(video_id)
        if fetched:
            segments = [{"text": s["text"], "start": s["start"], "end": s["start"] + s["duration"]} for s in fetched]
            logger.info("[extract] [video=%s] youtube-transcript-api got %d segments", video_id, len(segments))
            return segments
    except Exception as e:
        logger.warning("[extract] [video=%s] youtube-transcript-api failed: %s", video_id, e)

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


def _get_cookies_file() -> str | None:
    encoded = os.environ.get("YOUTUBE_COOKIES")
    if not encoded:
        return None
    import base64
    try:
        decoded = base64.b64decode(encoded).decode("utf-8")
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tmp.write(decoded)
        tmp.close()
        logger.info("[cookies] Wrote cookies file to %s (%d bytes)", tmp.name, len(decoded))
        return tmp.name
    except Exception as e:
        logger.warning("[cookies] Failed to decode YOUTUBE_COOKIES: %s", e)
        return None


def _extract_audio(video_url: str) -> str:
    logger.info("[audio] [url=%s] Extracting audio via yt-dlp...", video_url)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    yt_dlp_path = os.environ.get("YT_DLP_PATH", "yt-dlp")
    logger.info("[audio] Using yt-dlp path: %s", yt_dlp_path)

    cmd = [
        yt_dlp_path,
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", tmp.name.replace(".wav", ".%(ext)s"),
        "--print", "filename",
        "--no-check-certificates",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "--add-header", "Accept:text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "--add-header", "Accept-Language:en-US,en;q=0.9",
        "--add-header", "Origin:https://www.youtube.com",
        "--extractor-args", "youtube:player_client=android,web;skip=webpage",
        "--retries", "5",
        "--throttled-rate", "100M",
        "--concurrent-fragments", "1",
        "--geo-bypass",
    ]

    cookies_file = _get_cookies_file()
    if cookies_file:
        cmd.extend(["--cookies", cookies_file])
        logger.info("[audio] Using cookies file for authentication")

    cmd.append(video_url)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        if cookies_file and os.path.exists(cookies_file):
            os.unlink(cookies_file)
            logger.info("[audio] Cleaned up cookies file")

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
