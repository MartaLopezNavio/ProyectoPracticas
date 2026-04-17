import os
import json
import time
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, Response
from landmarks_unified import LandmarksEngine
from pose_server_discovery import start_discovery_server

OUTPUT_DIR = "mobile_output"
LATEST_FRAME_PATH = os.path.join(OUTPUT_DIR, "latest_frame.jpg")
LATEST_LANDMARKS_PATH = os.path.join(OUTPUT_DIR, "latest_landmarks.json")

app = Flask(__name__)
engine = LandmarksEngine(device="cuda:0", thr=0.3, face_thr=0.5)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_landmarks_json(landmarks, keypoints, scores, orientation, orientation_debug=None):
    def point_or_none(p):
        if p is None:
            return None
        return {
            "x": float(p[0]),
            "y": float(p[1])
        }

    payload = {
        "valid": landmarks is not None,
        "orientation": orientation,
        "measurement_allowed": False,
        "neck_base": None,
        "pelvis": None,
        "thyroid": None,
        "prostate": None,
        "keypoints": [],
        "orientation_debug": orientation_debug if orientation_debug is not None else {}
    }

    if landmarks is not None:
        payload["measurement_allowed"] = bool(
            landmarks.get("measurement_allowed", False)
        )
        payload["neck_base"] = point_or_none(landmarks.get("neck_base"))
        payload["pelvis"] = point_or_none(landmarks.get("pelvis"))
        payload["thyroid"] = point_or_none(landmarks.get("thyroid"))
        payload["prostate"] = point_or_none(landmarks.get("prostate"))

    if keypoints is not None and scores is not None:
        for i, (kp, sc) in enumerate(zip(keypoints, scores)):
            payload["keypoints"].append({
                "id": i,
                "x": float(kp[0]),
                "y": float(kp[1]),
                "score": float(sc)
            })

    with open(LATEST_LANDMARKS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    try:
        jpg_bytes = request.get_data()
        if not jpg_bytes:
            return jsonify({"error": "empty body"}), 400

        arr = np.frombuffer(jpg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "invalid jpeg"}), 400

        result = engine.process_frame(frame)

        final_img = result["image"]
        landmarks = result["landmarks"]
        keypoints = result["keypoints"]
        scores = result["scores"]
        orientation = result.get("orientation", "not_front")
        orientation_debug = result.get("orientation_debug", {})

        cv2.imwrite(LATEST_FRAME_PATH, final_img)
        save_landmarks_json(
            landmarks,
            keypoints,
            scores,
            orientation,
            orientation_debug=orientation_debug
        )

        return jsonify({
            "ok": True,
            "orientation": orientation,
            "measurement_allowed": bool(
                landmarks.get("measurement_allowed", False)
            ) if landmarks is not None else False,
            "orientation_debug": orientation_debug
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/frame.jpg", methods=["GET"])
def frame():
    if not os.path.exists(LATEST_FRAME_PATH):
        return jsonify({"error": "no frame yet"}), 404
    return send_file(LATEST_FRAME_PATH, mimetype="image/jpeg")


@app.route("/landmarks", methods=["GET"])
def landmarks():
    if not os.path.exists(LATEST_LANDMARKS_PATH):
        return jsonify({"error": "no landmarks yet"}), 404

    with open(LATEST_LANDMARKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return jsonify(data)


def mjpeg_generator():
    last_mtime = None

    while True:
        if not os.path.exists(LATEST_FRAME_PATH):
            time.sleep(0.05)
            continue

        try:
            current_mtime = os.path.getmtime(LATEST_FRAME_PATH)

            if last_mtime is None or current_mtime != last_mtime:
                with open(LATEST_FRAME_PATH, "rb") as f:
                    frame_bytes = f.read()

                last_mtime = current_mtime

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n" +
                    frame_bytes + b"\r\n"
                )
            else:
                time.sleep(0.03)

        except Exception:
            time.sleep(0.05)


@app.route("/stream.mjpg", methods=["GET"])
def stream():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "service": "mobile_pose_server"
    })


if __name__ == "__main__":
    start_discovery_server()
    app.run(host="0.0.0.0", port=8090, debug=False, threaded=True)
