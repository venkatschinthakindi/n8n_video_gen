from flask import Flask, request, jsonify, send_from_directory
import subprocess
import os
import uuid
import json

import urllib.request
from html.parser import HTMLParser
import random

app = Flask(__name__)

# HTML parser class
class ImgParser(HTMLParser):
    def __init__(self, search_text):
        super().__init__()
        self.search_text = search_text.lower()
        self.matched = []

    def handle_starttag(self, tag, attrs):
        if tag != "img":
            return
        attr_dict = dict(attrs)
        alt = attr_dict.get("alt", "")
        title_attr = attr_dict.get("title", "")
        src = attr_dict.get("src", "")
        if not src:
            return
        if self.search_text in alt.lower() or self.search_text in title_attr.lower():
            self.matched.append(src)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "text-to-video API is running"})

@app.route("/generate", methods=["POST"])
def generate_video():

    data = request.get_json(force=True)
    # Expect `texts` and `images` arrays in request JSON
    fileName = data.get("file_name")
    texts = data.get("texts")
    images = data.get("images")
    thumbnail = data.get("thumbnail")
    output_path = f"/app/output/{fileName}.mp4"
    thumbnail_file = f"/app/output/{fileName}.png"
    thumbnail_text = data.get("thumbnail_text")
    bg_music_file = data.get("bg_music_file", "")
    # Write text into a temp file to be read by text_to_video.py
    if not bg_music_file:
        bg_music_file = f"/app/bg_music{random.randint(1, 3)}.mp3"

     # Create the payload that the video generator expects
    payload = {
        "texts": texts,
        "images": images,
        "voice_file": f"/app/output/{fileName}.wav",
        "video_file": output_path,
        "thumbnail": thumbnail,
        "thumbnail_file": thumbnail_file,
        "thumbnail_text": thumbnail_text,
        "bg_music_file": bg_music_file
    }

    # Pass the payload JSON string via environment variable
    
    env = os.environ.copy()
    env["VIDEO_PAYLOAD"] = json.dumps(payload)
    env["VIDEO_FILE"] = output_path
    # env = os.environ.copy()

    # print(f"Generating video for request {request_id}...")

    result = subprocess.run(
        ["python", "text_to_video_ef.py"],
        capture_output=True,
        text=True,
        env=env,
        cwd="/app"
    )

    if result.returncode != 0:
        return jsonify({"error": "Video generation failed", "details": result.stderr}), 500

    return jsonify({
        "message": "Video created successfully",
        "output": env["VIDEO_FILE"]
    })

@app.route("/files", methods=["GET"])
def serve_video():
    try:
        filePath = request.args.get("path")
        # Sanitize filename to prevent directory traversal
        safe_filename = os.path.basename(filePath)
        file_path = os.path.join("/app/output",safe_filename)

        if not os.path.exists(file_path):
            return jsonify({"error": f"File '{safe_filename}' not found"}), 404

        # Serve the video file with proper headers
        return send_from_directory("/app/output", safe_filename, mimetype="video/mp4", as_attachment=False)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/get_thumbnail", methods=["GET"])
def serve_thumbnail():
    try:
        filePath = request.args.get("path")
        # Sanitize filename to prevent directory traversal
        safe_filename = os.path.basename(filePath)
        file_path = os.path.join("/app/output",safe_filename.replace(".mp4", ".png"))

        if not os.path.exists(file_path):
            return jsonify({"error": f"File '{safe_filename}' not found"}), 404

        # Serve the video file with proper headers
        return send_from_directory("/app/output", safe_filename, mimetype="image/jpeg", as_attachment=False)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/files", methods=["DELETE"])
def delete_file():
    try:
        file_path = request.args.get("path")
        if not file_path:
            return jsonify({"error": "File path is required"}), 400

        safe_filename = os.path.basename(file_path)

        # Full path (adjust 'base_dir' to your file storage location)
        base_dir = "/app/output"
        full_path = os.path.join(base_dir, safe_filename)

        if not os.path.exists(full_path):
            return jsonify({"error": "File not found"}), 404

        os.remove(full_path)
        os.remove(full_path.replace(".mp4", ".wav"))
        return jsonify({"message": f"File '{file_path}' deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API endpoint
@app.route("/extract-images", methods=["POST"])
def extract_images_endpoint():
    """
    Expects JSON payload:
    {
        "link": "https://example.com/article.html",
        "title": "keyword"
    }
    Returns JSON:
    {
        "images": ["url1", "url2", ...]
    }
    """
    data = request.get_json()
    link = data.get("link")
    title = data.get("title")

    if not link or not title:
        return jsonify({"error": "Missing 'link' or 'title' parameter"}), 400

    try:
        with urllib.request.urlopen(link) as response:
            html_content = response.read().decode("utf-8", errors="ignore")

        parser = ImgParser(title)
        parser.feed(html_content)
        return jsonify({"images": parser.matched})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
