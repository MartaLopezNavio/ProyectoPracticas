import argparse
import base64
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch


def get_depth_at_point(depth_map, x, y, window=5):
    h, w = depth_map.shape

    x = int(round(float(x)))
    y = int(round(float(y)))

    if x < 0 or x >= w or y < 0 or y >= h:
        return None

    half = window // 2

    x0 = max(0, x - half)
    x1 = min(w, x + half + 1)
    y0 = max(0, y - half)
    y1 = min(h, y + half + 1)

    patch = depth_map[y0:y1, x0:x1]
    vals = patch[np.isfinite(patch)]

    if vals.size == 0:
        return None

    vals = vals.astype(np.float32)
    vals = vals[vals > 0]

    if vals.size == 0:
        return None

    p20 = np.percentile(vals, 20)
    p80 = np.percentile(vals, 80)

    filtered = vals[(vals >= p20) & (vals <= p80)]

    if filtered.size == 0:
        return None

    return float(np.median(filtered))


def get_depths_at_points(depth_map, points, window=5):
    depths = {}
    errors = {}

    if not isinstance(points, dict):
        return depths, {"general": "points must be a dictionary"}

    for name, point in points.items():
        try:
            if point is None or not isinstance(point, dict):
                depths[name] = None
                errors[name] = "invalid point"
                continue

            x = point.get("x", None)
            y = point.get("y", None)

            if x is None or y is None:
                depths[name] = None
                errors[name] = "missing x or y"
                continue

            depth_value = get_depth_at_point(
                depth_map=depth_map,
                x=x,
                y=y,
                window=window,
            )

            depths[name] = depth_value
            errors[name] = None

        except Exception as e:
            depths[name] = None
            errors[name] = str(e)

    return depths, errors


class DepthAnythingServerModel:
    def __init__(self, model_name, device):
        from depth_anything_3.api import DepthAnything3

        if "cuda" in device and not torch.cuda.is_available():
            print("[WARN] CUDA no disponible en depth3. Uso CPU.")
            device = "cpu"

        self.device = torch.device(device)
        self.tmp_dir = tempfile.TemporaryDirectory()

        print(f"[INFO] Cargando Depth Anything 3: {model_name}")
        print(f"[INFO] Device: {self.device}")

        self.model = DepthAnything3.from_pretrained(model_name)
        self.model = self.model.to(self.device)

        if hasattr(self.model, "eval"):
            self.model.eval()

    def predict_depth(self, frame_bgr):
        """
        Recibe imagen BGR de OpenCV.
        Devuelve depth_map 2D alineado con la imagen.
        """

        tmp_path = os.path.join(self.tmp_dir.name, "current_frame.jpg")
        cv2.imwrite(tmp_path, frame_bgr)

        with torch.inference_mode():
            prediction = self.model.inference([tmp_path])

        depth = self.extract_depth(prediction)

        frame_h, frame_w = frame_bgr.shape[:2]

        if depth.shape != (frame_h, frame_w):
            depth = cv2.resize(
                depth,
                (frame_w, frame_h),
                interpolation=cv2.INTER_LINEAR,
            )

        return depth.astype(np.float32)

    @staticmethod
    def extract_depth(prediction):
        """
        Extrae el mapa depth de forma robusta.
        """

        depth = None

        if hasattr(prediction, "depth"):
            depth = prediction.depth
        elif isinstance(prediction, dict) and "depth" in prediction:
            depth = prediction["depth"]
        elif isinstance(prediction, (list, tuple)) and len(prediction) > 0:
            first = prediction[0]

            if hasattr(first, "depth"):
                depth = first.depth
            elif isinstance(first, dict) and "depth" in first:
                depth = first["depth"]

        if depth is None:
            raise RuntimeError("No se encontró el campo 'depth' en la predicción de DA3.")

        if hasattr(depth, "detach"):
            depth = depth.detach().cpu().numpy()

        depth = np.asarray(depth)

        if depth.ndim == 3:
            depth = depth[0]

        if depth.ndim != 2:
            raise RuntimeError(f"Shape de depth no esperado: {depth.shape}")

        return depth.astype(np.float32)


class Handler(BaseHTTPRequestHandler):
    model = None

    def _send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_payload(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if not body:
            raise RuntimeError("Cuerpo vacío.")

        return json.loads(body.decode("utf-8"))

    def _decode_image(self, image_b64):
        image_bytes = base64.b64decode(image_b64)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            raise RuntimeError("No se pudo decodificar la imagen recibida.")

        return frame_bgr

    def do_GET(self):
        if self.path == "/health":
            self._send_json({
                "ok": True,
                "message": "DA3 server running",
                "endpoints": [
                    "POST /depth_at_point",
                    "POST /depth_at_points",
                ],
            })
        else:
            self._send_json({
                "ok": False,
                "error": "Endpoint no encontrado",
            }, status=404)

    def do_POST(self):
        if self.path == "/depth_at_point":
            self.handle_depth_at_point()
            return

        if self.path == "/depth_at_points":
            self.handle_depth_at_points()
            return

        self._send_json({
            "ok": False,
            "error": "Endpoint no encontrado",
        }, status=404)

    def handle_depth_at_point(self):
        try:
            payload = self._read_payload()

            image_b64 = payload["image_jpg_b64"]
            x = float(payload["x"])
            y = float(payload["y"])
            window = int(payload.get("window", 5))

            frame_bgr = self._decode_image(image_b64)

            depth_map = self.model.predict_depth(frame_bgr)
            depth_value = get_depth_at_point(
                depth_map=depth_map,
                x=x,
                y=y,
                window=window,
            )

            self._send_json({
                "ok": True,
                "depth": depth_value,
                "depth_shape": list(depth_map.shape),
            })

        except Exception as e:
            self._send_json({
                "ok": False,
                "error": str(e),
            }, status=500)

    def handle_depth_at_points(self):
        try:
            payload = self._read_payload()

            image_b64 = payload["image_jpg_b64"]
            points = payload.get("points", {})
            window = int(payload.get("window", 5))

            if not isinstance(points, dict) or len(points) == 0:
                self._send_json({
                    "ok": False,
                    "error": "points vacío o inválido",
                }, status=400)
                return

            frame_bgr = self._decode_image(image_b64)

            # IMPORTANTE:
            # Aquí se ejecuta DA3 una sola vez.
            depth_map = self.model.predict_depth(frame_bgr)

            # Luego se extrae profundidad en varios puntos.
            depths, errors = get_depths_at_points(
                depth_map=depth_map,
                points=points,
                window=window,
            )

            self._send_json({
                "ok": True,
                "depths": depths,
                "errors": errors,
                "depth_shape": list(depth_map.shape),
                "window": window,
            })

        except Exception as e:
            self._send_json({
                "ok": False,
                "error": str(e),
            }, status=500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="depth-anything/DA3Metric-Large")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    Handler.model = DepthAnythingServerModel(
        model_name=args.model,
        device=args.device,
    )

    server = ThreadingHTTPServer((args.host, args.port), Handler)

    print(f"[INFO] Servidor DA3 activo en http://{args.host}:{args.port}")
    print("[INFO] Endpoint: POST /depth_at_point")
    print("[INFO] Endpoint: POST /depth_at_points")
    print("[INFO] Prueba de salud: http://127.0.0.1:8765/health")

    server.serve_forever()


if __name__ == "__main__":
    main()
