import whisper
import tempfile
import os
import subprocess
import re
from typing import List, Dict
from youtube_transcript_api import YouTubeTranscriptApi


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


def extract_youtube_transcript(video_url: str) -> List[Dict] | None:
    video_id = _extract_video_id(video_url)
    if not video_id:
        return None

    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=["en"])
        segments = []
        for snippet in fetched.snippets:
            segments.append({
                "text": snippet.text.strip(),
                "start": snippet.start,
                "end": snippet.start + snippet.duration,
            })
        return segments
    except Exception:
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
