import whisper
import tempfile
import os
import subprocess
from typing import List, Dict


_model = None


def get_model():
    global _model
    if _model is None:
        _model = whisper.load_model("base")
    return _model


def extract_audio(video_url: str) -> str:
    """Download audio from a URL using yt-dlp and return path to WAV file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    yt_dlp_path = os.environ.get("YT_DLP_PATH", "yt-dlp")
    ffmpeg_path = os.environ.get("FFMPEG_PATH", "ffmpeg")

    result = subprocess.run(
        [
            yt_dlp_path,
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", tmp.name.replace(".wav", ".%(ext)s"),
            "--print", "filename",
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

    # yt-dlp adds the extension automatically, find the actual file
    base = tmp.name.replace(".wav", "")
    for ext in [".wav", ".mp3", ".m4a", ".webm"]:
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Could not find downloaded audio file ({tmp.name})")


def transcribe(audio_path: str) -> List[Dict]:
    """Transcribe audio file using Whisper and return timestamped segments."""
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
    """Full pipeline: download audio, transcribe, cleanup."""
    audio_path = None
    try:
        audio_path = extract_audio(video_url)
        segments = transcribe(audio_path)
        full_text = " ".join(s["text"] for s in segments)
        return {
            "segments": segments,
            "full_text": full_text,
            "language": "en",
        }
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
