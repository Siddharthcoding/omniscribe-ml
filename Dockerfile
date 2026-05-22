FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    openssl \
    curl \
    && update-ca-certificates --fresh \
    && apt-get upgrade -y --no-install-recommends openssl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir --retries 5 --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir --retries 5 -r requirements.txt

COPY . .

ENV YT_DLP_PATH=yt-dlp
ENV FFMPEG_PATH=ffmpeg
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
