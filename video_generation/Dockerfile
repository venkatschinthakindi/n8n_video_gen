FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    espeak  \
    libespeak1 \
    ffmpeg \
    espeak-ng \
    libespeak-ng1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt
# RUN pip install moviepy pyttsx3 TTS pillow numpy flask

COPY . .
RUN chmod -R 777 /app

EXPOSE 8080
ENV PYTHONUNBUFFERED=1
ENV TTS_SKIP_PROMPTS=1
ENV TTS_SKIP_PROMPTS=1 \
    COQUI_TOS_AGREED=1 \
    COQUI_STUDIO_TOS=1

CMD ["python", "server.py"]
