from flask import Flask, request, jsonify, send_from_directory
import subprocess
import os
import uuid
import json

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "text-to-video API is running"})

@app.route("/generate", methods=["POST"])
def generate_video():

    request_id = str(uuid.uuid4())[:8]
    data = request.get_json(force=True)
    # Expect `texts` and `images` arrays in request JSON
    texts = data.get("texts")
    images = data.get("images")
    thumbnail = data.get("thumbnail")
    output_path = f"/app/output/{request_id}_news_video.mp4"

    # Write text into a temp file to be read by text_to_video.py
   

     # Create the payload that the video generator expects
    payload = {
        "texts": texts,
        "images": images,
        "voice_file": f"/app/output/{request_id}_voice.wav",
        "video_file": output_path,
        "thumbnail": thumbnail
    }

    # Pass the payload JSON string via environment variable
    
    env = os.environ.copy()
    env["VIDEO_PAYLOAD"] = json.dumps(payload)
    env["VIDEO_FILE"] = output_path
    # env = os.environ.copy()

    # print(f"Generating video for request {request_id}...")

    result = subprocess.run(
        ["python", "text_to_video.py"],
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
        os.remove(full_path.replace("_news_video.mp4", "_voice.wav"))
        return jsonify({"message": f"File '{file_path}' deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
