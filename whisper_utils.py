import whisper
import tempfile
import os
import subprocess
import re
from typing import List, Dict


_model = None


def get_model():
    global _model
    if _model is None:
        _model = whisper.load_model("tiny")
    return _model


def extract_subtitles(video_url: str) -> List[Dict] | None:
    tmp_dir = tempfile.mkdtemp()
    yt_dlp_path = os.environ.get("YT_DLP_PATH", "yt-dlp")

    subprocess.run(
        [
            yt_dlp_path,
            "--skip-download",
            "--write-auto-subs",
            "--sub-lang", "en",
            "-o", os.path.join(tmp_dir, "%(id)s"),
            video_url,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    sub_path = None
    for f in os.listdir(tmp_dir):
        if f.endswith(".vtt") or f.endswith(".srt"):
            sub_path = os.path.join(tmp_dir, f)
            break

    if not sub_path or not os.path.exists(sub_path):
        return None

    with open(sub_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    segments = _parse_vtt(content) if sub_path.endswith(".vtt") else _parse_srt(content)

    for f in os.listdir(tmp_dir):
        fp = os.path.join(tmp_dir, f)
        try:
            os.remove(fp)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return segments if segments else None


def _parse_vtt(content: str) -> List[Dict]:
    segments = []
    block_pattern = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{3})\n([\s\S]*?)(?=\n\n|\Z)",
        re.MULTILINE,
    )

    def _ts_to_seconds(ts: str) -> float:
        ts = ts.replace(",", ".")
        parts = ts.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return 0.0

    for match in block_pattern.finditer(content):
        start = _ts_to_seconds(match.group(1))
        end = _ts_to_seconds(match.group(2))
        text = " ".join(match.group(3).strip().split("\n"))
        text = re.sub(r"<[^>]+>", "", text)
        if text:
            segments.append({"text": text, "start": start, "end": end})

    return segments


def _parse_srt(content: str) -> List[Dict]:
    segments = []
    block_pattern = re.compile(
        r"\d+\n(\d{1,2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{3})\n([\s\S]*?)(?=\n\n|\Z)",
        re.MULTILINE,
    )

    def _ts_to_seconds(ts: str) -> float:
        ts = ts.replace(",", ".")
        parts = ts.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return 0.0

    for match in block_pattern.finditer(content):
        start = _ts_to_seconds(match.group(1))
        end = _ts_to_seconds(match.group(2))
        text = " ".join(match.group(3).strip().split("\n"))
        if text:
            segments.append({"text": text, "start": start, "end": end})

    return segments


def extract_audio(video_url: str) -> str:
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
    segments = extract_subtitles(video_url)
    if segments:
        full_text = " ".join(s["text"] for s in segments)
        return {
            "segments": segments,
            "full_text": full_text,
            "language": "en",
            "source": "subtitles",
        }

    audio_path = None
    try:
        audio_path = extract_audio(video_url)
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
