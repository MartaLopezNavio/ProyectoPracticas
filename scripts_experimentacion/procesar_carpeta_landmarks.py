#!/usr/bin/env python3
"""Procesa una carpeta de imágenes y guarda landmarks y visualizaciones."""

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPTS_DIR = PROJECT_ROOT / "scripts_principales"
if str(MAIN_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_SCRIPTS_DIR))

from landmarks_unified import LandmarksEngine

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def point_or_none(point):
    if point is None:
        return None
    return {"x": float(point[0]), "y": float(point[1])}


def main():
    parser = argparse.ArgumentParser(description="Procesamiento offline de pose y landmarks.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--body-thr", type=float, default=0.3)
    parser.add_argument("--face-thr", type=float, default=0.5)
    args = parser.parse_args()

    images = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"No se encontraron imágenes en {args.input_dir}")

    annotated_dir = args.output_dir / "imagenes_anotadas"
    json_dir = args.output_dir / "landmarks_json"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    engine = LandmarksEngine(
        device=args.device,
        thr=args.body_thr,
        face_thr=args.face_thr,
        front_frames_required=1,
        not_front_frames_required=1,
    )

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] No se pudo leer {image_path}")
            continue

        result = engine.process_frame(image)
        cv2.imwrite(str(annotated_dir / image_path.name), result["image"])

        landmarks = result.get("landmarks") or {}
        payload = {
            "source_image": image_path.name,
            "success": bool(result.get("success", False)),
            "orientation": result.get("orientation", "not_front"),
            "measurement_allowed": bool(landmarks.get("measurement_allowed", False)),
            "landmarks": {
                "neck_base": point_or_none(landmarks.get("neck_base")),
                "thyroid": point_or_none(landmarks.get("thyroid")),
                "pelvis": point_or_none(landmarks.get("pelvis")),
                "prostate": point_or_none(landmarks.get("prostate")),
            },
            "orientation_debug": result.get("orientation_debug", {}),
        }
        (json_dir / f"{image_path.stem}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"Resultados guardados en {args.output_dir}")


if __name__ == "__main__":
    main()
