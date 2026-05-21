FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel
RUN pip install --no-cache-dir --retries 5 --no-build-isolation setuptools==67.8.0
RUN pip install --no-cache-dir --retries 5 --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir --retries 5 --no-build-isolation -r requirements.txt

COPY . .

ENV YT_DLP_PATH=yt-dlp
ENV FFMPEG_PATH=ffmpeg

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
