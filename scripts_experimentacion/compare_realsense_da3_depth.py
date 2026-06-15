import os
import csv
import json
import time
import base64
import argparse
import urllib.request
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPTS_DIR = PROJECT_ROOT / "scripts_principales"
if str(MAIN_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_SCRIPTS_DIR))

from landmarks_unified import LandmarksEngine


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

COLOR_WIDTH = 640
COLOR_HEIGHT = 480
FPS = 15

DEPTH_TARGETS = ["thyroid", "prostate"]

BODY_THR = 0.3
FACE_THR = 0.5

DA3_SERVER_URL = "http://127.0.0.1:8765/depth_at_points"

DEFAULT_DEPTH_WINDOW = 25
DEFAULT_DA3_EVERY_N_FRAMES = 3

REAL_SENSE_MIN_DEPTH = 0.2
REAL_SENSE_MAX_DEPTH = 5.0


# ============================================================
# CLIENTE DEPTH ANYTHING 3
# ============================================================

class DepthAnythingClient:
    def __init__(self, url, timeout=60):
        self.url = url
        self.timeout = timeout

    def get_depths_at_points(self, frame_bgr, points, window=25):
        ok, buffer = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 70],
        )

        if not ok:
            raise RuntimeError("No se pudo codificar el frame como JPG.")

        image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        payload = {
            "image_jpg_b64": image_b64,
            "points": points,
            "window": int(window),
        }

        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        if not data.get("ok", False):
            raise RuntimeError(data.get("error", "Error desconocido en servidor DA3."))

        return data.get("depths", {}), data.get("errors", {})


# ============================================================
# UTILIDADES
# ============================================================

def point_to_xy(point):
    if point is None:
        return None

    if isinstance(point, dict):
        if "x" in point and "y" in point:
            return float(point["x"]), float(point["y"])
        return None

    arr = np.asarray(point, dtype=np.float32).reshape(-1)

    if arr.size < 2:
        return None

    if not np.isfinite(arr[0]) or not np.isfinite(arr[1]):
        return None

    return float(arr[0]), float(arr[1])


def point_inside_image(x, y):
    return 0 <= x < COLOR_WIDTH and 0 <= y < COLOR_HEIGHT


def collect_available_points(landmarks):
    points = {}

    if landmarks is None:
        return points

    for target_name in DEPTH_TARGETS:
        xy = point_to_xy(landmarks.get(target_name))

        if xy is None:
            continue

        x, y = xy

        if not point_inside_image(x, y):
            continue

        points[target_name] = {
            "x": float(x),
            "y": float(y),
        }

    return points


def get_realsense_depth_at_pixel(depth_frame, x, y, window=25):
    """
    Obtiene una profundidad robusta de RealSense alrededor del landmark.
    Devuelve:
    - depth_m
    - valid_count
    - local_std
    """

    if depth_frame is None:
        return None, 0, None

    width = depth_frame.get_width()
    height = depth_frame.get_height()

    x = int(round(float(x)))
    y = int(round(float(y)))

    if x < 0 or x >= width or y < 0 or y >= height:
        return None, 0, None

    half = window // 2
    values = []

    for yy in range(max(0, y - half), min(height, y + half + 1)):
        for xx in range(max(0, x - half), min(width, x + half + 1)):
            d = depth_frame.get_distance(xx, yy)

            if d is not None and np.isfinite(d):
                if REAL_SENSE_MIN_DEPTH < d < REAL_SENSE_MAX_DEPTH:
                    values.append(float(d))

    if len(values) == 0:
        return None, 0, None

    values = np.array(values, dtype=np.float32)

    # Filtro robusto para quitar extremos
    p20 = np.percentile(values, 20)
    p80 = np.percentile(values, 80)
    filtered = values[(values >= p20) & (values <= p80)]

    if filtered.size == 0:
        return None, int(values.size), None

    depth_m = float(np.median(filtered))
    local_std = float(np.std(filtered))

    return depth_m, int(filtered.size), local_std


def safe_float(value):
    if value is None:
        return ""
    try:
        if not np.isfinite(float(value)):
            return ""
        return float(value)
    except Exception:
        return ""


def compute_abs_error(value, real_distance):
    if value is None or real_distance is None:
        return None
    return abs(float(value) - float(real_distance))


def draw_text(img, text, pos, color=(255, 255, 255), scale=0.48, thickness=1):
    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_overlay(img, orientation, rows_for_frame):
    out = img.copy()

    if orientation == "front":
        orientation_color = (0, 255, 0)
    else:
        orientation_color = (0, 0, 255)

    draw_text(out, f"orient: {orientation}", (20, 32), orientation_color)

    y = 58

    for row in rows_for_frame:
        target = row["target"]
        rs = row["realsense_depth_m"]
        da3 = row["da3_depth_m"]

        rs_txt = f"{rs:.3f}" if rs is not None else "None"
        da3_txt = f"{da3:.3f}" if da3 is not None else "None"

        draw_text(
            out,
            f"{target} | RS: {rs_txt} m | DA3: {da3_txt} m",
            (20, y),
            (255, 255, 255),
        )

        y += 24

    return out


# ============================================================
# MÉTRICAS
# ============================================================

def summarize_metrics(rows, real_distance):
    """
    Calcula métricas simples por target y método.
    """

    summary = []

    for target in DEPTH_TARGETS:
        target_rows = [r for r in rows if r["target"] == target]

        for method_name, key in [
            ("realsense", "realsense_depth_m"),
            ("depth_anything3", "da3_depth_m"),
        ]:
            values = [
                float(r[key])
                for r in target_rows
                if r[key] is not None and np.isfinite(float(r[key]))
            ]

            total_rows = len(target_rows)
            valid_count = len(values)
            valid_rate = valid_count / total_rows if total_rows > 0 else 0.0

            if valid_count > 0:
                arr = np.array(values, dtype=np.float32)

                mean_depth = float(np.mean(arr))
                std_depth = float(np.std(arr))

                if arr.size >= 2:
                    frame_to_frame_diff = float(np.mean(np.abs(np.diff(arr))))
                else:
                    frame_to_frame_diff = None

                if real_distance is not None:
                    reference = float(real_distance)
                    errors = np.abs(arr - reference)
                    mae = float(np.mean(errors))
                    rmse = float(np.sqrt(np.mean(errors ** 2)))
                    mape = (
                        float(np.mean(errors / abs(reference)) * 100.0)
                        if reference != 0.0
                        else None
                    )
                else:
                    mae = None
                    rmse = None
                    mape = None
            else:
                mean_depth = None
                std_depth = None
                frame_to_frame_diff = None
                mae = None
                rmse = None
                mape = None

            summary.append({
                "target": target,
                "method": method_name,
                "total_rows": total_rows,
                "valid_count": valid_count,
                "valid_rate": valid_rate,
                "mean_depth_m": mean_depth,
                "std_depth_m": std_depth,
                "mean_frame_to_frame_diff_m": frame_to_frame_diff,
                "real_distance_m": real_distance,
                "mae_m": mae,
                "rmse_m": rmse,
                "mape_percent": mape,
            })

    return summary


def save_summary_csv(path, summary):
    fieldnames = [
        "target",
        "method",
        "total_rows",
        "valid_count",
        "valid_rate",
        "mean_depth_m",
        "std_depth_m",
        "mean_frame_to_frame_diff_m",
        "real_distance_m",
        "mae_m",
        "rmse_m",
        "mape_percent",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary:
            writer.writerow({
                key: safe_float(row.get(key)) if isinstance(row.get(key), (float, int, type(None))) else row.get(key)
                for key in fieldnames
            })


def print_summary(summary):
    print("\n================ RESUMEN MÉTRICAS ================")

    for row in summary:
        print(
            f"{row['target']} | {row['method']} | "
            f"valid={row['valid_count']}/{row['total_rows']} "
            f"({row['valid_rate']:.2f}) | "
            f"mean={row['mean_depth_m']} | "
            f"std={row['std_depth_m']} | "
            f"diff_frame={row['mean_frame_to_frame_diff_m']} | "
            f"MAE={row['mae_m']} | "
            f"RMSE={row['rmse_m']} | MAPE={row['mape_percent']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comparación RealSense depth vs Depth Anything 3 en landmarks anatómicos."
    )

    parser.add_argument(
        "--real-distance",
        type=float,
        default=None,
        help="Distancia real en metros medida manualmente. Ejemplo: 1.2",
    )

    parser.add_argument(
        "--label",
        type=str,
        default="test",
        help="Etiqueta de la prueba. Ejemplo: cerca_1_2m",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "resultados" / "depth_comparison"),
        help="Carpeta donde guardar CSV y resumen.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duración máxima de la prueba en segundos.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Número máximo de frames a guardar.",
    )

    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Frames iniciales que no se guardan para dejar estabilizar cámara/modelo.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Dispositivo para RTMPose: cuda:0 o cpu.",
    )

    parser.add_argument(
        "--da3-every-n",
        type=int,
        default=DEFAULT_DA3_EVERY_N_FRAMES,
        help="Frecuencia de consulta a DA3. 1 = todos los frames.",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_DEPTH_WINDOW,
        help="Ventana alrededor del landmark para calcular profundidad.",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No mostrar ventana de OpenCV.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_{args.label}"

    csv_path = os.path.join(args.output_dir, f"{base_name}_frames.csv")
    summary_path = os.path.join(args.output_dir, f"{base_name}_summary.csv")
    latest_frame_path = os.path.join(args.output_dir, f"{base_name}_last_frame.jpg")

    print("======================================")
    print("Comparación RealSense vs Depth Anything 3")
    print(f"CSV frames: {csv_path}")
    print(f"CSV resumen: {summary_path}")
    print(f"Distancia real: {args.real_distance}")
    print(f"DA3 endpoint: {DA3_SERVER_URL}")
    print(f"DA3 cada {args.da3_every_n} frame(s)")
    print("======================================")

    print("[INFO] Cargando RTMPose...")
    engine = LandmarksEngine(
        device=args.device,
        thr=BODY_THR,
        face_thr=FACE_THR,
    )

    da3_client = DepthAnythingClient(DA3_SERVER_URL)

    da3_cache = {
        "thyroid": None,
        "prostate": None,
    }

    da3_errors = {
        "thyroid": None,
        "prostate": None,
    }

    da3_fresh = {
        "thyroid": False,
        "prostate": False,
    }

    # RealSense
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        COLOR_WIDTH,
        COLOR_HEIGHT,
        rs.format.bgr8,
        FPS,
    )

    config.enable_stream(
        rs.stream.depth,
        COLOR_WIDTH,
        COLOR_HEIGHT,
        rs.format.z16,
        FPS,
    )

    print("[INFO] Iniciando RealSense RGB + Depth...")
    pipeline.start(config)

    align = rs.align(rs.stream.color)

    rows = []
    saved_frames = 0
    frame_idx = 0
    start_time = time.time()

    fieldnames = [
        "frame_idx",
        "time_s",
        "test_label",
        "real_distance_m",
        "orientation",
        "measurement_allowed",
        "target",
        "landmark_available",
        "target_x",
        "target_y",
        "realsense_depth_m",
        "realsense_valid_count",
        "realsense_local_std_m",
        "da3_depth_m",
        "da3_is_fresh",
        "da3_error",
        "abs_error_realsense_m",
        "abs_error_da3_m",
    ]

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                now = time.time()
                elapsed = now - start_time

                if args.duration is not None and elapsed >= args.duration:
                    break

                if saved_frames >= args.max_frames:
                    break

                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                frame_bgr = np.asanyarray(color_frame.get_data())

                result = engine.process_frame(frame_bgr)

                final_img = result["image"]
                landmarks = result["landmarks"]
                orientation = result.get("orientation", "not_front")

                measurement_allowed = False

                if landmarks is not None:
                    measurement_allowed = bool(
                        landmarks.get("measurement_allowed", False)
                    ) and orientation == "front"

                points = collect_available_points(landmarks)

                # Consultar DA3 una sola vez para todos los puntos disponibles
                for name in DEPTH_TARGETS:
                    da3_fresh[name] = False

                should_query_da3 = (
                    measurement_allowed
                    and len(points) > 0
                    and (
                        frame_idx % args.da3_every_n == 0
                        or all(da3_cache[name] is None for name in DEPTH_TARGETS)
                    )
                )

                if should_query_da3:
                    try:
                        depths, errors = da3_client.get_depths_at_points(
                            frame_bgr=frame_bgr,
                            points=points,
                            window=args.window,
                        )

                        for name in DEPTH_TARGETS:
                            if name in points:
                                value = depths.get(name, None)

                                da3_cache[name] = float(value) if value is not None else None
                                da3_errors[name] = errors.get(name, None) if isinstance(errors, dict) else None
                                da3_fresh[name] = True
                            else:
                                da3_cache[name] = None
                                da3_errors[name] = "point_missing_or_outside"
                                da3_fresh[name] = True

                    except Exception as e:
                        for name in DEPTH_TARGETS:
                            da3_errors[name] = str(e)
                            da3_fresh[name] = True

                rows_for_frame = []

                for target_name in DEPTH_TARGETS:
                    xy = None

                    if landmarks is not None:
                        xy = point_to_xy(landmarks.get(target_name))

                    landmark_available = False
                    x = None
                    y = None

                    if xy is not None:
                        x, y = xy
                        landmark_available = point_inside_image(x, y)

                    realsense_depth = None
                    realsense_valid_count = 0
                    realsense_local_std = None

                    if measurement_allowed and landmark_available:
                        realsense_depth, realsense_valid_count, realsense_local_std = (
                            get_realsense_depth_at_pixel(
                                depth_frame=depth_frame,
                                x=x,
                                y=y,
                                window=args.window,
                            )
                        )

                    da3_depth = None
                    da3_error = None
                    da3_is_fresh = False

                    if measurement_allowed and landmark_available:
                        da3_depth = da3_cache.get(target_name, None)
                        da3_error = da3_errors.get(target_name, None)
                        da3_is_fresh = da3_fresh.get(target_name, False)

                    abs_error_rs = compute_abs_error(
                        realsense_depth,
                        args.real_distance,
                    )

                    abs_error_da3 = compute_abs_error(
                        da3_depth,
                        args.real_distance,
                    )

                    row = {
                        "frame_idx": frame_idx,
                        "time_s": elapsed,
                        "test_label": args.label,
                        "real_distance_m": args.real_distance,
                        "orientation": orientation,
                        "measurement_allowed": measurement_allowed,
                        "target": target_name,
                        "landmark_available": landmark_available,
                        "target_x": x,
                        "target_y": y,
                        "realsense_depth_m": realsense_depth,
                        "realsense_valid_count": realsense_valid_count,
                        "realsense_local_std_m": realsense_local_std,
                        "da3_depth_m": da3_depth,
                        "da3_is_fresh": da3_is_fresh,
                        "da3_error": da3_error,
                        "abs_error_realsense_m": abs_error_rs,
                        "abs_error_da3_m": abs_error_da3,
                    }

                    writer.writerow({
                        key: safe_float(row[key]) if key not in [
                            "test_label",
                            "orientation",
                            "target",
                            "da3_error",
                        ] else row[key]
                        for key in fieldnames
                    })

                    rows.append(row)
                    rows_for_frame.append(row)

                if frame_idx >= args.warmup_frames:
                    saved_frames += 1

                final_img = draw_overlay(
                    img=final_img,
                    orientation=orientation,
                    rows_for_frame=rows_for_frame,
                )

                cv2.imwrite(latest_frame_path, final_img)

                if not args.no_show:
                    cv2.imshow("Comparacion RealSense vs DA3", final_img)

                    key = cv2.waitKey(1) & 0xFF

                    if key == 27:
                        break

                frame_idx += 1

    except KeyboardInterrupt:
        pass

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    summary = summarize_metrics(
        rows=rows,
        real_distance=args.real_distance,
    )

    save_summary_csv(
        path=summary_path,
        summary=summary,
    )

    print_summary(summary)

    print("\n[OK] Comparación terminada.")
    print(f"[OK] CSV frames: {csv_path}")
    print(f"[OK] CSV resumen: {summary_path}")
    print(f"[OK] Último frame: {latest_frame_path}")


if __name__ == "__main__":
    main()
