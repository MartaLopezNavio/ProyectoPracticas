#!/usr/bin/env python3
"""Consulta el servidor DA3 para imágenes y landmarks procesados offline."""

import argparse
import base64
import csv
import json
import urllib.request
from pathlib import Path

import cv2

TARGETS = ("thyroid", "prostate")


def request_depths(url, image, points, window, timeout):
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen.")
    payload = {
        "image_jpg_b64": base64.b64encode(buffer.tobytes()).decode("utf-8"),
        "points": points,
        "window": int(window),
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not data.get("ok", False):
        raise RuntimeError(data.get("error", "Error desconocido en DA3"))
    return data.get("depths", {}), data.get("errors", {})


def main():
    parser = argparse.ArgumentParser(description="Extracción offline de profundidad en landmarks.")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--landmarks-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8765/depth_at_points")
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for json_path in sorted(args.landmarks_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_name = data.get("source_image") or f"{json_path.stem}.jpg"
        image_path = args.images_dir / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] No se pudo leer {image_path}")
            continue

        landmarks = data.get("landmarks", data)
        points = {
            name: landmarks[name]
            for name in TARGETS
            if isinstance(landmarks.get(name), dict)
            and "x" in landmarks[name]
            and "y" in landmarks[name]
        }
        if not points:
            continue

        try:
            depths, errors = request_depths(
                args.url, image, points, args.window, args.timeout
            )
        except Exception as exc:
            depths, errors = {}, {name: str(exc) for name in points}

        for name, point in points.items():
            rows.append({
                "image": image_name,
                "target": name,
                "x": point["x"],
                "y": point["y"],
                "depth": depths.get(name),
                "error": errors.get(name),
                "orientation": data.get("orientation"),
                "measurement_allowed": data.get("measurement_allowed"),
            })

    fieldnames = [
        "image", "target", "x", "y", "depth", "error",
        "orientation", "measurement_allowed"
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV guardado en {args.output_csv} ({len(rows)} filas)")


if __name__ == "__main__":
    main()
