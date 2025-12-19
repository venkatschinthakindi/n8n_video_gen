# ---------- Dockerfile ----------
FROM python:3.11-slim

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    TTS_SKIP_PROMPTS=true

# Minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install TTS==0.11.0
# Install Coqui TTS + torch (CPU-only)
RUN pip install  torch==2.3.0 soundfile numpy


# Copy code and models
COPY v_telugu.py /app/v_telugu.py
COPY models /app/models

# Default run command
CMD ["python3", "v_telugu.py"]
