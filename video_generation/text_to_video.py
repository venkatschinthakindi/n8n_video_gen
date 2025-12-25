import os
import json
import requests
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
    CompositeAudioClip,
)
from TTS.api import TTS
import textwrap
from moviepy.audio.AudioClip import AudioArrayClip
import re
# -----------------------------
# 1. Read input payload (from env or file)
# -----------------------------
def parse_image_urls(image_urls_string):
    if not image_urls_string:
        return []

    # Remove any surrounding quotes or whitespace
    cleaned = image_urls_string.strip().strip('"').strip("'")

    # Split on commas OR just before each 'http'
    parts = re.split(r',\s*|(?=https?://)', cleaned)

    # Clean up and validate
    urls = [
        p.strip()
        for p in parts
        if p.strip().startswith(("http://", "https://"))
    ]

    return urls

payload_str = os.getenv("VIDEO_PAYLOAD")
if not payload_str:
    raise ValueError("VIDEO_PAYLOAD environment variable not found")

try:
    payload = json.loads(payload_str)
except json.JSONDecodeError:
    raise ValueError("Invalid JSON in VIDEO_PAYLOAD")

texts = payload.get("texts", [])
image_urls_string = payload.get("images", "")
voice_file = payload.get("voice_file", "")
video_file = payload.get("video_file", "")
video_size = tuple(payload.get("video_size", [720, 1280]))
bg_music_file = payload.get("bg_music_file", "/app/bg_music.mp3")  # optional
thumbnail = payload.get("thumbnail", "")

if not texts or not image_urls_string:
    raise ValueError("Payload must include 'texts' and 'images' arrays")

image_urls = parse_image_urls(image_urls_string)
if len(thumbnail) > 8:
    image_urls.insert(0, thumbnail)

os.makedirs(os.path.dirname(video_file), exist_ok=True)

# -----------------------------
# 2. Generate TTS Audio
# -----------------------------
tts_text = " ".join(texts)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
tts.tts_to_file(text=tts_text, file_path=voice_file, speed=0.9)

if not os.path.exists(voice_file):
    raise RuntimeError("TTS audio file was not created!")

audio_clip = AudioFileClip(voice_file)
video_duration = audio_clip.duration

# -----------------------------
# 3. Download and prepare images
# -----------------------------
segment_duration = video_duration / len(image_urls)
image_clips = []

for idx, url in enumerate(image_urls):
    try:
        response = requests.get(url, timeout=15)  # set a timeout for reliability

        if response.status_code != 200:
            continue

        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_np = np.array(img)

        clip = (
            ImageClip(img_np)
            .resized(video_size)               # note: correct method is .resize(), not .resized()
            .with_duration(segment_duration)   # correct method: .set_duration()
        )

        image_clips.append(clip)

    except Exception as e:
        print(f"[ERROR] Failed to process {url}: {e}")
        continue

background_clip = concatenate_videoclips(image_clips, method="compose")

# -----------------------------
# 4. Create scrolling text
# -----------------------------
scroll_text = "\n\n".join(texts)
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font = ImageFont.truetype(font_path, 30)
lines = textwrap.wrap(scroll_text, width=35)

img_width, img_height = 1200, 100 * len(lines)
img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)
y_offset = 0
for line in lines:
    bbox = draw.textbbox((0, 0), line, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img_width - w) / 2
    draw.text((x, y_offset), line, font=font, fill="white")
    y_offset += h + 10

scroll_img_clip = ImageClip(np.array(img)).with_duration(video_duration)


# Calculate total scroll distance
total_scroll = img_height + video_size[1]

# Calculate scroll speed (pixels per second)
scroll_speed = total_scroll / (video_duration * 2)



# Function to return position at time t
def scroll_position(t):
    y = video_size[1] - scroll_speed * t
    return ("center", y)

# Apply dynamic scrolling to the clip
scroll_clip = scroll_img_clip.with_position(scroll_position)


# -----------------------------
# 5. Add background music (looped to match video)
# -----------------------------
if os.path.exists(bg_music_file):
    bg_music_clip = AudioFileClip(bg_music_file)
    bg_fps = 44100
    bg_array = bg_music_clip.to_soundarray(fps=bg_fps) * 0.12  # reduce volume

    # Loop BGM to match video duration
    loops = int(video_duration // bg_music_clip.duration) + 1
    bg_array_full = np.tile(bg_array, (loops, 1))
    samples_needed = int(video_duration * bg_fps)
    bg_array_full = bg_array_full[:samples_needed]

    bg_music_loop = AudioArrayClip(bg_array_full, fps=bg_fps).with_duration(video_duration)

    final_audio = CompositeAudioClip([audio_clip, bg_music_loop]).with_duration(video_duration)
else:
    final_audio = audio_clip

# -----------------------------
# 6. Combine video + audio
# -----------------------------
final_clip = CompositeVideoClip([background_clip, scroll_clip]).with_audio(final_audio)
final_clip.write_videofile(video_file, fps=24)

print(f"✅ Video created successfully with background music: {video_file}")
