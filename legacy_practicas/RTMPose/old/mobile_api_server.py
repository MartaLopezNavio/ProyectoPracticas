import os
import json
import time
from flask import Flask, Response, send_file, jsonify

OUTPUT_DIR = "mobile_output"
LATEST_FRAME_PATH = os.path.join(OUTPUT_DIR, "latest_frame.jpg")
LATEST_LANDMARKS_PATH = os.path.join(OUTPUT_DIR, "latest_landmarks.json")

app = Flask(__name__)


def file_exists(path):
    return os.path.exists(path) and os.path.isfile(path)


@app.route("/")
def index():
    return jsonify({
        "status": "ok",
        "endpoints": {
            "frame": "/frame.jpg",
            "stream": "/stream.mjpg",
            "landmarks": "/landmarks",
            "status": "/status"
        }
    })


@app.route("/status")
def status():
    return jsonify({
        "frame_available": file_exists(LATEST_FRAME_PATH),
        "landmarks_available": file_exists(LATEST_LANDMARKS_PATH),
        "frame_path": LATEST_FRAME_PATH,
        "landmarks_path": LATEST_LANDMARKS_PATH
    })


@app.route("/frame.jpg")
def frame():
    if not file_exists(LATEST_FRAME_PATH):
        return jsonify({"error": "No frame available"}), 404
    return send_file(LATEST_FRAME_PATH, mimetype="image/jpeg")


@app.route("/landmarks")
def landmarks():
    if not file_exists(LATEST_LANDMARKS_PATH):
        return jsonify({"error": "No landmarks available"}), 404

    try:
        with open(LATEST_LANDMARKS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Failed to read landmarks: {e}"}), 500


def mjpeg_generator():
    last_mtime = None

    while True:
        if not file_exists(LATEST_FRAME_PATH):
            time.sleep(0.1)
            continue

        try:
            current_mtime = os.path.getmtime(LATEST_FRAME_PATH)

            if last_mtime is None or current_mtime != last_mtime:
                with open(LATEST_FRAME_PATH, "rb") as f:
                    frame = f.read()

                last_mtime = current_mtime

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
                    frame + b"\r\n"
                )
            else:
                time.sleep(0.03)

        except Exception:
            time.sleep(0.1)


@app.route("/stream.mjpg")
def stream():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
