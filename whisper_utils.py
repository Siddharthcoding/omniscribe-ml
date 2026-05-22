import whisper
import tempfile
import os
import subprocess
import re
import httpx
from typing import List, Dict


_model = None


def get_model():
    global _model
    if _model is None:
        _model = whisper.load_model("tiny")
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


def _decode_html(text: str) -> str:
    return text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'").replace("&quot;", '"')


def _parse_transcript_xml(xml: str) -> List[Dict] | None:
    segments = []

    for match in re.finditer(r'<text start="([\d.]+)" dur="([\d.]+)">(.*?)</text>', xml):
        text = _decode_html(match.group(3))
        start = float(match.group(1))
        dur = float(match.group(2))
        segments.append({"text": text, "start": start, "end": start + dur})

    if not segments:
        for match in re.finditer(r'<p t="([\d.]+)" d="([\d.]+)"[^>]*>(.*?)</p>', xml):
            content = re.sub(r"<[^>]+>", "", match.group(3))
            text = _decode_html(content)
            t = float(match.group(1)) / 1000
            d = float(match.group(2)) / 1000
            segments.append({"text": text, "start": t, "end": t + d})

    return segments if segments else None


def _fetch_transcript_from_tracks(caption_tracks: List[Dict]) -> List[Dict] | None:
    track = None
    for t in caption_tracks:
        if t.get("languageCode") == "en":
            track = t
            break
    if not track and caption_tracks:
        track = caption_tracks[0]
    if not track:
        return None

    try:
        resp = httpx.get(track["baseUrl"], timeout=10)
        if resp.status_code != 200:
            return None
        return _parse_transcript_xml(resp.text)
    except Exception:
        return None


def extract_youtube_transcript(video_url: str) -> List[Dict] | None:
    video_id = _extract_video_id(video_url)
    if not video_id:
        return None

    try:
        resp = httpx.post(
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
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            caption_tracks = data.get("captions", {}).get("playerCaptionsTracklistRenderer", {}).get("captionTracks", [])
            if caption_tracks:
                result = _fetch_transcript_from_tracks(caption_tracks)
                if result:
                    return result
    except Exception:
        pass

    try:
        resp = httpx.get(
            f"https://youtubetranscript.com/?v={video_id}",
            timeout=10,
        )
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
            return segments
    except Exception:
        pass

    return None


def _extract_audio(video_url: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    yt_dlp_path = os.environ.get("YT_DLP_PATH", "yt-dlp")

    result = subprocess.run(
        [
            yt_dlp_path,
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", tmp.name.replace(".wav", ".%(ext)s"),
            "--print", "filename",
            "--no-check-certificates",
            "--extractor-args", "youtube:player_client=android",
            "--extractor-args", "youtube:skip=webpage",
            video_url,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (exit {result.returncode}):\n"
            f"stderr: {result.stderr[:2000]}"
        )

    actual_file = result.stdout.strip()
    if actual_file and os.path.exists(actual_file):
        return actual_file

    base = tmp.name.replace(".wav", "")
    for ext in [".wav", ".mp3", ".m4a", ".webm"]:
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Could not find downloaded audio file ({tmp.name})")


def transcribe(audio_path: str) -> List[Dict]:
    model = get_model()
    result = model.transcribe(audio_path, word_timestamps=True)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "text": seg["text"].strip(),
            "start": seg["start"],
            "end": seg["end"],
        })

    return segments


def transcribe_from_url(video_url: str) -> dict:
    segments = extract_youtube_transcript(video_url)
    if segments:
        full_text = " ".join(s["text"] for s in segments)
        return {
            "segments": segments,
            "full_text": full_text,
            "language": "en",
            "source": "youtube_api",
        }

    audio_path = None
    try:
        audio_path = _extract_audio(video_url)
        segments = transcribe(audio_path)
        full_text = " ".join(s["text"] for s in segments)
        return {
            "segments": segments,
            "full_text": full_text,
            "language": "en",
            "source": "whisper",
        }
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
