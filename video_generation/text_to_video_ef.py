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
    AudioArrayClip,
    CompositeAudioClip,
    VideoClip,
    ColorClip,
    TextClip,
    concatenate_videoclips
)
from TTS.api import TTS
import cv2
import re
import textwrap
from pydub import AudioSegment

from PIL import Image, ImageDraw, ImageFont
import textwrap

from PIL import Image, ImageDraw, ImageFont
import textwrap

from PIL import Image, ImageDraw, ImageFont
import hashlib

def create_thumbnail_from_array(
    images_np,
    text,
    output_path="thumbnail.png",
    size=(1080, 1920),
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-BoldItalic.ttf",
    first_line_font_size=160,
    other_lines_font_size= 100,
    first_line_color=(255, 255, 255, 255),   # White
    other_lines_color=(255, 165, 0, 255),    # Orange
    shadow_color=(0, 0, 0, 200),
    line_spacing=1.2
):
    if not images_np:
        raise ValueError("images_np is empty — no images to process.")

    # Base image
    img = Image.fromarray(images_np[0])
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    img_w, img_h = img.size
    bg = Image.new("RGBA", size, (0, 0, 0, 255))
    bg.paste(img, ((size[0]-img_w)//2, (size[1]-img_h)//2))
    img = bg

    # Text layer
    txt_layer = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(txt_layer)

    # Wrap text
    max_text_width = int(img_w * 0.9)
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        # Use first line font size temporarily to check width
        font_size = first_line_font_size if not lines else other_lines_font_size
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
        if draw.textlength(test_line, font=font) <= max_text_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    # Compute total height with variable font sizes
    line_heights = []
    for i, line in enumerate(lines):
        font_size = first_line_font_size if i == 0 else other_lines_font_size
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0,0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])
    
    total_height = sum(line_heights) + (len(lines)-1)*line_heights[0]*(line_spacing-1)
    y_cursor = (img_h - total_height)/2

    # Draw each line with its font size and color
    for i, line in enumerate(lines):
        font_size = first_line_font_size if i == 0 else other_lines_font_size
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
        color = first_line_color if i == 0 else other_lines_color
        x_cursor = (img_w - draw.textlength(line, font=font))/2

        # Shadow
        for dx in (-3,3):
            for dy in (-3,3):
                draw.text((x_cursor+dx, y_cursor+dy), line, font=font, fill=shadow_color)
        # Text
        draw.text((x_cursor, y_cursor), line, font=font, fill=color)
        y_cursor += line_heights[i] * line_spacing

    # Merge layers
    img = Image.alpha_composite(img, txt_layer)
    img.save(output_path, format="PNG", quality=95)
    return output_path

# -----------------------------
# 1. Read input payload
# -----------------------------
payload_str = os.getenv("VIDEO_PAYLOAD")
if not payload_str:
    raise ValueError("VIDEO_PAYLOAD environment variable not found")

payload = json.loads(payload_str)

texts = payload.get("texts", [])
thumbnail_url = payload.get("thumbnail", "")
voice_file = payload.get("voice_file", "voice.wav")
video_file = payload.get("video_file", "output.mp4")
thumbnail_file = payload.get("thumbnail_file", "thumbnail.jpg")
thumbnail_text = payload.get("thumbnail_text", "")
video_size = tuple(payload.get("video_size", [1080, 1920]))
bg_music_file = payload.get("bg_music_file", "/app/bg_music.mp3")  # optional

if not texts:
    raise ValueError("Payload must include 'texts' and 'thumbnail'")

os.makedirs(os.path.dirname(video_file), exist_ok=True)

# -----------------------------
# 2. Generate TTS Audio
# -----------------------------
reference_wav = "/app/reference_voice.wav"  # optional
tts_text = " ".join(texts)
tts = TTS(model_name="tts_models/en/vctk/vits", gpu=False)
tts.tts_to_file(text=tts_text, 
                file_path=voice_file,
                speaker_wav=reference_wav,
                speaker='p248')

# load and slow down
sound = AudioSegment.from_file(voice_file)
slower = sound._spawn(sound.raw_data, overrides={
    "frame_rate": int(sound.frame_rate * 0.9)  # 0.8 = slower, 1.2 = faster
}).set_frame_rate(sound.frame_rate)

slower.export(voice_file, format="wav")

audio_clip = AudioFileClip(voice_file)
video_duration = audio_clip.duration

image_urls_string = payload.get("images", "")
thumbnail_url = payload.get("thumbnail", "")
video_size = (1080, 1920)
max_images = 4

# -------------------------------------------------------------------
# PARSE AND CLEAN URL LIST
# -------------------------------------------------------------------
image_urls_tmp = re.split(r',(?=https?://)', image_urls_string)
seen = set()
image_urls = [
    clean for raw in image_urls_tmp
    if (clean := raw.strip()) and not (clean.lower() in seen or seen.add(clean.lower()))
]

# -------------------------------------------------------------------
# Sort image URLs by quality (Content-Length or dimension)
# -------------------------------------------------------------------
def estimate_quality(url):
    try:
        resp = requests.head(url, timeout=8, allow_redirects=True)
        size = int(resp.headers.get("Content-Length", 0))
        if size > 0:
            return size  # use size when available

        # fallback: use pixel area
        img_resp = requests.get(url, timeout=8, stream=True)
        img = Image.open(BytesIO(img_resp.content))
        width, height = img.size
        return width * height
    except Exception:
        return 0

# Sort descending (high-quality first)
image_urls.sort(key=estimate_quality, reverse=True)

# -------------------------------------------------------------------
# Prepare image array
# -------------------------------------------------------------------
images_np = []
if thumbnail_url:
    image_urls.insert(0, thumbnail_url)

# -------------------------------------------------------------------
# Fallback to static image if no URLs
# -------------------------------------------------------------------
if not image_urls:
    fallback_path = "/app/TrendFlicks.png"
    img = Image.open(fallback_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    if not video_size or not (
        isinstance(video_size, tuple)
        and len(video_size) == 2
        and all(isinstance(x, int) and x > 0 for x in video_size)
    ):
        raise ValueError(f"Invalid video_size: {video_size}")

    img = img.resize(video_size, Image.LANCZOS)
    images_np.append(np.array(img))
else:
    # ----------------------------------------------------------------
    # Download sorted images (unique by content hash)
    # ----------------------------------------------------------------
    downloaded_hashes = set()
    for idx, url in enumerate(image_urls):
        if len(images_np) >= max_images:
            print(f"✅ Reached {max_images} valid images, skipping remaining URLs.")
            break

        try:
            print(f"Downloading image {idx+1}/{len(image_urls)}: {url}")
            resp = requests.get(url, timeout=40)
            resp.raise_for_status()

            # Deduplicate by image hash
            img_hash = hashlib.md5(resp.content).hexdigest()
            if img_hash in downloaded_hashes:
                print(f"⚠️ Skipped duplicate image: {url}")
                continue
            downloaded_hashes.add(img_hash)

            img = Image.open(BytesIO(resp.content)).convert("RGB").resize(video_size, Image.LANCZOS)
            images_np.append(np.array(img))
            print(f"✅ Loaded image {len(images_np)} / {max_images}")
        except Exception as e:
            print(f"⚠️ Failed to load {url}: {e}")

# -------------------------------------------------------------------
# Fallback: Try thumbnail if all failed
# -------------------------------------------------------------------
if not images_np:
    print("⚠️ All images failed to download. Trying thumbnail as fallback.")
    try:
        resp = requests.get(thumbnail_url, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB").resize(video_size, Image.LANCZOS)
        images_np.append(np.array(img))
        print("✅ Thumbnail loaded as fallback.")
    except Exception as e:
        raise ValueError(f"❌ Failed to load even the thumbnail: {e}")

thumb_path = create_thumbnail_from_array(
    images_np=images_np,
    text=thumbnail_text,
    output_path=thumbnail_file,
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-BoldItalic.ttf"
)

# If fewer than max_images, loop to reach target for smooth animation
if len(images_np) < max_images:
    print(f"ℹ️ Only {len(images_np)} valid images loaded — looping to reach {max_images}.")
    repeat_count = (max_images // len(images_np)) + 1
    images_np = (images_np * repeat_count)[:max_images]

fps = 24
num_images = len(images_np)
total_duration = video_duration

# --- Adaptive Timing ---
fade_sec = 0.8                         # cross-fade overlap
min_image_time = 5.0                   # lower bound per image
max_image_time = 8.0                   # upper bound per image
raw_time = total_duration / num_images
image_time = max(min(raw_time, max_image_time), min_image_time)
effective_duration = image_time * num_images
last_extension = max(0.0, total_duration - effective_duration)

zoom_strength = 0.10
pan_strength =  0.06
print(f"Adaptive timing: {num_images} images → {image_time:.2f}s each (total {effective_duration:.1f}s)")

def make_frame(t):
    image_index = int(t // image_time) % num_images
    next_index = (image_index + 1) % num_images

    img_current = images_np[image_index].astype(np.float32) / 255.0
    img_next = images_np[next_index].astype(np.float32) / 255.0
    h, w = img_current.shape[:2]

    local_t = t % image_time
    progress = local_t / image_time

    # --- Zoom and Pan on current image ---
    zoom_factor = 1.0 + zoom_strength * progress
    M_zoom = cv2.getRotationMatrix2D((w / 2, h / 2), 0, zoom_factor)
    zoomed = cv2.warpAffine(img_current, M_zoom, (w, h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    direction = 1 if image_index % 2 == 0 else -1
    pan_x = int(direction * pan_strength * w * progress)
    pan_y = int(pan_strength * h * 0.2 * np.sin(progress * np.pi))
    M_pan = np.float32([[1, 0, pan_x], [0, 1, pan_y]])
    zoomed_panned = cv2.warpAffine(zoomed, M_pan, (w, h),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # --- Cross-fade logic ---
    if local_t >= image_time - fade_sec:
        # fade out current and fade in next
        fade_progress = (local_t - (image_time - fade_sec)) / fade_sec

        # apply zoom/pan for next image as well for continuity
        next_zoom_factor = 1.0 + zoom_strength * fade_progress
        M_next_zoom = cv2.getRotationMatrix2D((w / 2, h / 2), 0, next_zoom_factor)
        next_zoomed = cv2.warpAffine(img_next, M_next_zoom, (w, h),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        next_direction = 1 if next_index % 2 == 0 else -1
        pan_x_next = int(next_direction * pan_strength * w * fade_progress)
        pan_y_next = int(pan_strength * h * 0.2 * np.sin(fade_progress * np.pi))
        M_next_pan = np.float32([[1, 0, pan_x_next], [0, 1, pan_y_next]])
        next_zoomed_panned = cv2.warpAffine(next_zoomed, M_next_pan, (w, h),
                                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # blend
        frame = (1 - fade_progress) * zoomed_panned + fade_progress * next_zoomed_panned
    else:
        frame = zoomed_panned

    return np.clip(frame * 255, 0, 255).astype(np.uint8)

animated_clip = VideoClip(make_frame, duration=total_duration)

# -----------------------------
# 5. Scrolling text overlay (MoviePy + PIL)
# -----------------------------
scroll_text = tts_text.upper()  # or join multiple texts if you have more
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-Italic.ttf"
font = ImageFont.truetype(font_path, 35)

# Wrap the text for readability
lines = textwrap.wrap(scroll_text, width=40)

# Create a tall transparent image with all lines
img_w, img_h = video_size[0], 80 * len(lines)
text_img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
draw = ImageDraw.Draw(text_img)

y_offset = 0
for line in lines:
    bbox = draw.textbbox((0, 0), line, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img_w - w) / 2
    draw.text((x, y_offset), line, font=font, fill=(255, 165, 0, 255))
    y_offset += h + 20

# Convert the drawn text to an ImageClip
scroll_img_clip = ImageClip(np.array(text_img)).with_duration(video_duration)

# Calculate scroll distance and speed
total_scroll = img_h + video_size[1]
scroll_speed = (total_scroll / video_duration) * 0.6

# Define the scrolling position function
def scroll_position(t):
    # y starts below the bottom and moves upward over time
    y = video_size[1] - scroll_speed * t
    return ("center", y)

# Build the scroll clip
scroll_clip = (
    scroll_img_clip
    .with_position(scroll_position)
    .with_duration(video_duration)
    .with_opacity(1)
)

# Combine with background or base video
animated_text_clip = CompositeVideoClip(
    [animated_clip, scroll_clip],
    size=video_size
).with_duration(video_duration)


# -----------------------------
# 6. Add background music
# -----------------------------
final_audio = audio_clip
if os.path.exists(bg_music_file):
    bg_clip = AudioFileClip(bg_music_file)
    bg_fps = 44100
    bg_array = bg_clip.to_soundarray(fps=bg_fps) * 0.30
    loops = int(video_duration // bg_clip.duration) + 1
    bg_array_full = np.tile(bg_array, (loops, 1))
    samples_needed = int(video_duration * bg_fps)
    bg_array_full = bg_array_full[:samples_needed]
    bg_music_loop = AudioArrayClip(bg_array_full, fps=bg_fps).with_duration(video_duration)
    final_audio = CompositeAudioClip([audio_clip, bg_music_loop]).with_duration(video_duration)

# -----------------------------
# 7. Compose final video
# -----------------------------
final_width = max(animated_clip.w, video_size[0])
final_height = max(animated_clip.h, video_size[1])
background_clip = ColorClip(size=(final_width, final_height), color=(0, 0, 0), duration=video_duration)

animated_clip = animated_clip.with_position("center")
animated_text_clip = animated_text_clip.with_position("center")

if animated_clip.mask is None:
    animated_clip = animated_clip.with_mask(animated_clip.to_mask())
if animated_text_clip.mask is None:
    animated_text_clip = animated_text_clip.with_mask(animated_text_clip.to_mask())

final_clip = CompositeVideoClip([background_clip, animated_clip, animated_text_clip])
final_clip = final_clip.with_audio(final_audio)

# 3. Create outro ImageClip using that same path
outro_duration = 3  # seconds
outro_clip = ImageClip(thumb_path, duration=outro_duration)
outro_clip = outro_clip.resized(final_clip.size)

# Optional fade
# outro_clip = outro_clip.fadein(0.5).fadeout(0.5)
# final_clip = final_clip.fadeout(0.5)

# 4. Concatenate main video + outro
final_with_outro = concatenate_videoclips([outro_clip,final_clip])


final_with_outro.write_videofile(
    video_file,
    fps=fps,
    codec="libx264",
    audio_codec="aac",
    bitrate="5000k",
    preset="medium",
    threads=4
)


# final_clip.write_videofile(video_file, fps=fps,
#     codec="libx264",
#     audio_codec="aac",
#     bitrate="5000k",   # 5 Mbps for 1080p
#     preset="medium",
#     threads=4)

print(f"✅ Video created successfully with cinematic animation and TTS: {video_file}")
