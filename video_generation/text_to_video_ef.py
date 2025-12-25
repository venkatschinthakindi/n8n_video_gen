# import os
# import json
# import requests
# import numpy as np
# from io import BytesIO
# from PIL import Image, ImageDraw, ImageFont
# from moviepy import (
#     AudioFileClip,
#     ImageClip,
#     CompositeVideoClip,
#     AudioArrayClip,
#     CompositeAudioClip,
#     VideoClip,
#     ColorClip
# )
# from TTS.api import TTS
# import textwrap
# import cv2
# # -----------------------------
# # 1. Read input payload
# # -----------------------------
# payload_str = os.getenv("VIDEO_PAYLOAD")
# if not payload_str:
#     raise ValueError("VIDEO_PAYLOAD environment variable not found")

# payload = json.loads(payload_str)

# texts = payload.get("texts", [])
# thumbnail_url = payload.get("thumbnail", "")
# voice_file = payload.get("voice_file", "voice.wav")
# video_file = payload.get("video_file", "output.mp4")
# video_size = tuple(payload.get("video_size", [720, 1280]))
# bg_music_file = payload.get("bg_music_file", "/app/bg_music.mp3")  # optional

# if not texts or not thumbnail_url:
#     raise ValueError("Payload must include 'texts' and 'thumbnail'")

# os.makedirs(os.path.dirname(video_file), exist_ok=True)

# # -----------------------------
# # 2. Generate TTS Audio
# # -----------------------------
# tts_text = " ".join(texts)
# tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
# tts.tts_to_file(text=tts_text, file_path=voice_file, speed=0.9)
# audio_clip = AudioFileClip(voice_file)
# video_duration = audio_clip.duration

# # -----------------------------
# # 3. Download and prepare thumbnail image
# # -----------------------------
# response = requests.get(thumbnail_url, timeout=15)
# response.raise_for_status()
# img = Image.open(BytesIO(response.content)).convert("RGB")
# img_np = np.array(img)
# img_clip_base = ImageClip(img_np).resized(video_size)

# # -----------------------------
# # 4. Create timeline-based cinematic animation
# # -----------------------------
# fps = 15
# num_frames = int(video_duration * fps)

# # Segment durations in seconds
# fade_in_sec = 5
# zoom1_sec = 8
# shake_sec = 2
# rotate_sec = 3
# zoom2_sec = 8
# fade_out_sec = video_duration - (fade_in_sec + zoom1_sec + shake_sec + rotate_sec + zoom2_sec)

# fade_in_frames = int(fade_in_sec * fps)
# zoom1_frames = int(zoom1_sec * fps)
# shake_frames = int(shake_sec * fps)
# rotate_frames = int(rotate_sec * fps)
# zoom2_frames = int(zoom2_sec * fps)
# fade_out_frames = int(fade_out_sec * fps)

# max_zoom = 2.15
# shake_intensity = 3
# max_rotation = 360

# # Function to generate frame transformations
# def make_frame(t):
#     i = int(t * fps)
#     frame = img_np.copy()              # use base numpy image instead of ImageClip frame
#     h, w = frame.shape[:2]

#     def rotate_and_crop(image, angle, scale=1.0, shake_x=0, shake_y=0):
#         M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
#         M[0, 2] += shake_x
#         M[1, 2] += shake_y
#         return cv2.warpAffine(image, M, (w, h),
#                               flags=cv2.INTER_LINEAR,
#                               borderMode=cv2.BORDER_REFLECT)

#     if i < fade_in_frames:  # Fade-in
#         alpha = i / fade_in_frames
#         frame_mod = cv2.convertScaleAbs(frame * alpha)

#     elif i < fade_in_frames + zoom1_frames:  # First zoom
#         phase = (i - fade_in_frames) / zoom1_frames
#         zoom = 1.0 + (max_zoom - 1.0) * (phase / 0.5 if phase <= 0.5 else 1 - (phase - 0.5) / 0.5)
#         frame_mod = rotate_and_crop(frame, 0, zoom)

#     elif i < fade_in_frames + zoom1_frames + shake_frames:  # Shake
#         shake_x = np.random.randint(-shake_intensity, shake_intensity + 1)
#         shake_y = np.random.randint(-shake_intensity, shake_intensity + 1)
#         frame_mod = rotate_and_crop(frame, 0, 1.0, shake_x, shake_y)

#     elif i < fade_in_frames + zoom1_frames + shake_frames + rotate_frames:  # Rotate 360°
#         phase = (i - fade_in_frames - zoom1_frames - shake_frames) / rotate_frames
#         angle = max_rotation * phase
#         frame_mod = rotate_and_crop(frame, angle, 1.0)

#     elif i < fade_in_frames + zoom1_frames + shake_frames + rotate_frames + zoom2_frames:  # Second zoom
#         phase = (i - fade_in_frames - zoom1_frames - shake_frames - rotate_frames) / zoom2_frames
#         zoom = 1.0 + (max_zoom - 1.0) * (phase / 0.5 if phase <= 0.5 else 1 - (phase - 0.5) / 0.5)
#         frame_mod = rotate_and_crop(frame, 0, zoom)

#     else:  # Fade-out
#         alpha = max(0, (num_frames - i) / fade_out_frames)
#         frame_mod = cv2.convertScaleAbs(frame * alpha)

#     return frame_mod

# # ✅ Correct way in MoviePy 2.x — use VideoClip instead of ImageClip.with_make_frame
# animated_clip = VideoClip(make_frame, duration=video_duration)

# # -----------------------------
# # 5. Create scrolling text overlay
# # -----------------------------
# scroll_text = "\n\n".join(texts)
# font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# font = ImageFont.truetype(font_path, 30)
# lines = textwrap.wrap(scroll_text, width=35)
# img_w, img_h = 1200, 100*len(lines)
# text_img = Image.new("RGBA", (img_w, img_h), (0,0,0,0))
# draw = ImageDraw.Draw(text_img)
# y_offset = 0
# for line in lines:
#     bbox = draw.textbbox((0,0), line, font=font)
#     w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
#     x = (img_w - w)/2
#     draw.text((x, y_offset), line, font=font, fill="white")
#     y_offset += h+10

# scroll_img_clip = ImageClip(np.array(text_img)).with_duration(video_duration)

# total_scroll = img_h + video_size[1]
# scroll_speed = total_scroll / (video_duration*2)

# # def scroll_position(t):
# #     y = video_size[1] - scroll_speed*t
# #     return ("center", y)

# # def scroll_position(t):
# #     y = video_size[1] - scroll_speed * t
# #     # Clamp Y so it never goes far offscreen
# #     if y < -video_size[1]:
# #         y = -video_size[1]
# #     return ("center", y)

            
# # # scroll_clip = scroll_img_clip.with_duration(scroll_position)
# # scroll_clip = scroll_img_clip.with_duration(video_duration).with_position(scroll_position)

# # --- Safe scrolling text setup ---
# def scroll_position(t):
#     y = video_size[1] - scroll_speed * t
#     # Clamp Y so text never leaves the render area (prevents 0-height mask)
#     if y < -video_size[1] + 5:
#         y = -video_size[1] + 5
#     return ("center", y)

# scroll_clip = (
#     scroll_img_clip
#     .with_position(video_duration)
#     .with_position(scroll_position)
#     .resized(video_size)           # Correct for ImageClip
#     .with_opacity(1)
# )


# # -----------------------------
# # 6. Add background music
# # -----------------------------
# final_audio = audio_clip
# if os.path.exists(bg_music_file):
#     bg_clip = AudioFileClip(bg_music_file)
#     bg_fps = 44100
#     bg_array = bg_clip.to_soundarray(fps=bg_fps)*0.12
#     loops = int(video_duration//bg_clip.duration)+1
#     bg_array_full = np.tile(bg_array, (loops,1))
#     samples_needed = int(video_duration*bg_fps)
#     bg_array_full = bg_array_full[:samples_needed]
#     bg_music_loop = AudioArrayClip(bg_array_full, fps=bg_fps).with_duration(video_duration)
#     final_audio = CompositeAudioClip([audio_clip, bg_music_loop]).with_duration(video_duration)

# # -----------------------------
# # 7. Combine animated image, text, and audio
# # -----------------------------
# # final_clip = CompositeVideoClip([animated_clip, scroll_clip]).with_audio(final_audio)
# # Determine final video size (use largest width and height among clips)
# final_width = max(animated_clip.w, scroll_clip.w)
# final_height = max(animated_clip.h, scroll_clip.h)

# # Optionally, create a black background clip to fill empty space
# background_clip = ColorClip(
#     size=(final_width, final_height),
#     color=(0, 0, 0),
#     duration=max(animated_clip.duration, scroll_clip.duration)
# )

# # Center scroll_clip and animated_clip on background
# animated_clip = animated_clip.with_position("center")
# scroll_clip = scroll_clip.with_position("center")

# # Remove mask issues by ensuring all clips have proper masks
# if animated_clip.mask is None:
#     animated_clip = animated_clip.with_mask(animated_clip.to_mask())
# if scroll_clip.mask is None:
#     scroll_clip = scroll_clip.with_mask(scroll_clip.to_mask())

# # Compose final clip
# final_clip = CompositeVideoClip([background_clip, animated_clip, scroll_clip])

# # Add audio if needed
# final_clip = final_clip.with_audio(final_audio)

# final_clip.write_videofile(video_file, fps=fps)

# print(f"✅ Video created successfully with cinematic animation and TTS: {video_file}")


#4 multiple images with effects (google)
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
    TextClip
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

def create_thumbnail_from_array(
    images_np,
    text,
    output_path="thumbnail.png",
    size=(1080, 1920),
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    font_size=60,
    text_color=(255, 165, 0, 255),     # orange with alpha
    shadow_color=(0, 0, 0, 180),       # semi-transparent shadow
    position="center",
    max_width_ratio=0.9,               # 90% of image width
    max_height_ratio=0.8,              # 80% of image height
    line_spacing=1.2                   # space between lines
):
    """
    Creates a thumbnail from the first image in a NumPy array.
    Auto-wraps and scales text to fit the image, adds soft shadow, and saves as PNG.
    """
    if not images_np:
        raise ValueError("images_np is empty — no images to process.")

    # Prepare base image
    img = Image.fromarray(images_np[0])
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    img_w, img_h = img.size

    # Create transparent text layer
    txt_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_layer)

    # Load font with fallback
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # --- Auto-fit: reduce font size until text fits width & height ---
    while True:
        # Wrap text within width
        max_text_width = int(img_w * max_width_ratio)
        lines = []
        for line in text.split("\n"):
            wrapped = textwrap.wrap(line, width=40)
            lines.extend(wrapped if wrapped else [""])

        # Measure wrapped text block
        line_heights, line_widths = [], []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            line_widths.append(w)
            line_heights.append(h)

        total_height = sum(line_heights) + (len(lines) - 1) * (line_heights[0] * (line_spacing - 1))
        text_width = max(line_widths) if line_widths else 0
        text_height = total_height

        if text_width <= max_text_width and text_height <= img_h * max_height_ratio:
            break  # fits nicely
        font_size -= 2
        if font_size < 12:
            break  # prevent infinite loop if text is extremely long
        font = ImageFont.truetype(font_path, font_size)

    # --- Positioning ---
    if position == "top":
        y = img_h * 0.1
    elif position == "bottom":
        y = img_h - text_height - 50
    else:  # center
        y = (img_h - text_height) / 2
    x = (img_w - text_width) / 2

    # --- Draw text with shadow ---
    shadow_offset = 2
    y_cursor = y
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        x_line = (img_w - line_width) / 2

        # Draw shadow
        for dx in (-shadow_offset, shadow_offset):
            for dy in (-shadow_offset, shadow_offset):
                draw.text((x_line + dx, y_cursor + dy), line, font=font, fill=shadow_color)

        # Draw text
        draw.text((x_line, y_cursor), line, font=font, fill=text_color)
        y_cursor += line_heights[i] * line_spacing

    # Merge text layer with image
    img = Image.alpha_composite(img, txt_layer)

    # Save
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
tts_text = " ".join(texts)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
tts.tts_to_file(text=tts_text, file_path=voice_file)

# load and slow down
sound = AudioSegment.from_file(voice_file)
slower = sound._spawn(sound.raw_data, overrides={
    "frame_rate": int(sound.frame_rate * 0.9)  # 0.8 = slower, 1.2 = faster
}).set_frame_rate(sound.frame_rate)

slower.export(voice_file, format="wav")

audio_clip = AudioFileClip(voice_file)
video_duration = audio_clip.duration

# -----------------------------
# 3. Adaptive Multi-Image Cinematic Sequence (Zoom + Pan + Cross-Fade)
# -----------------------------
image_urls_string = payload.get("images", "")
# other_images = [u.strip() for u in image_urls_string.split(",") if u.strip()]
image_urls_tmp = re.split(r',(?=https?://)', image_urls_string)
other_images = [url.strip() for url in image_urls_tmp]  # clean extra spaces/newlines

if thumbnail_url.startswith("http"):
    image_urls = [thumbnail_url] + other_images  # Thumbnail first
else:
    image_urls = other_images

image_urls = [u for u in image_urls if u.strip()]
images_np = []

if not image_urls or len(image_urls) == 0:
    fallback_path = "/app/TrendFlicks.png"
    img = Image.open(fallback_path)
    # Convert only if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

     # Ensure video_size is defined and valid
    if "video_size" not in locals() or not video_size:
        video_size = (1280, 720)  # fallback default
    if not (isinstance(video_size, tuple)
            and len(video_size) == 2
            and all(isinstance(x, int) and x > 0 for x in video_size)):
        raise ValueError(f"Invalid video_size: {video_size}")

    # Resize safely
    img = img.resize(video_size, Image.LANCZOS)

    # Append to list
    images_np.append(np.array(img))
# -----------------------------
# Image Download (stop after 6 valid images)
# -----------------------------

max_images = 4  # ideal number for 30–40 second videos

for idx, url in enumerate(image_urls):
    if len(images_np) >= max_images:
        print(f"✅ Reached {max_images} valid images, skipping remaining URLs.")
        break

    try:
        print(f"Downloading image {idx+1}/{len(image_urls)}: {url}")
        resp = requests.get(url, timeout=40)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB").resize(video_size, Image.LANCZOS)
        images_np.append(np.array(img))
        print(f"✅ Loaded image {len(images_np)} / {max_images}")
    except Exception as e:
        print(f"⚠️ Failed to load {url}: {e}")

# Fallback if no images
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

# thumb_path = create_thumbnail_from_array(
#     images_np=images_np,
#     text=thumbnail_text,
#     output_path=thumbnail_file,
#     font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # or any .ttf font
#     font_size=25,
#     position="center"
# )

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
# 5. Word-by-word popping text overlay
# -----------------------------
# words = tts_text.upper().split()
# highlight_words = {"IMPORTANT", "ACTION", "VICTORY", "WIN", "BAD", "VIOLANCE"}
# word_duration = video_duration / max(len(words), 1)
# text_clips = []

# for idx, word in enumerate(words):
#     start_time = idx * word_duration
#     end_time = start_time + word_duration
#     if idx % 3 == 0 or word.strip(".,!?:;") in highlight_words:
#         fontsize = 90
#         color = "orange"
#     else:
#         fontsize = 70
#         color = "white"

#     txt_clip = TextClip(
#         text=word,
#         size=(1200, 1200),
#         font="/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-BoldItalic.ttf",
#         color=color,
#         font_size=fontsize,
#         method="caption"
#     ).with_start(start_time).with_end(end_time).with_position(("center", "center"))

#     text_clips.append(txt_clip)

# animated_text_clip = CompositeVideoClip(text_clips, size=video_size).with_duration(video_duration)


# -----------------------------
# 5. Scrolling text overlay (MoviePy + PIL)
# -----------------------------
scroll_text = tts_text.upper()  # or join multiple texts if you have more
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-Italic.ttf"
font = ImageFont.truetype(font_path, 30)

# Wrap the text for readability
lines = textwrap.wrap(scroll_text, width=32)

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
final_clip.write_videofile(video_file, fps=fps,
    codec="libx264",
    audio_codec="aac",
    bitrate="5000k",   # 5 Mbps for 1080p
    preset="medium",
    threads=4)

print(f"✅ Video created successfully with cinematic animation and TTS: {video_file}")
