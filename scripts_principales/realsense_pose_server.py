import os
import json
import base64
import urllib.request
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
import pyrealsense2 as rs

from landmarks_unified import LandmarksEngine


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "mobile_output"
LATEST_FRAME_PATH = OUTPUT_DIR / "latest_frame.jpg"
LATEST_LANDMARKS_PATH = OUTPUT_DIR / "latest_landmarks.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RealSense SOLO COMO CÁMARA RGB
COLOR_WIDTH = 640
COLOR_HEIGHT = 480
FPS = 15

# RTMPose
DEVICE = "cuda:0"
BODY_THR = 0.3
FACE_THR = 0.5


# ============================================================
# DEPTH ANYTHING 3
# ============================================================

DA3_SERVER_URL = "http://127.0.0.1:8765/depth_at_points"

DEPTH_TARGETS = ["thyroid", "prostate"]

# Primero usa tiroides. Si no hay tiroides, usa próstata.
# Si no hay ninguna, devuelve SIN_MEDIDA.
PRIMARY_TARGET_PRIORITY = ["thyroid", "prostate"]

DEPTH_WINDOW = 25

DA3_EVERY_N_FRAMES = 3

SMOOTH_LEN = 5
STATE_LEN = 5


# ============================================================
# LÓGICA DE PROXIMIDAD CON DEPTH ANYTHING 3
# ============================================================

# IMPORTANTE:
# Estos valores NO se interpretan como metros reales.
# Son valores relativos devueltos por Depth Anything 3.
#
# depth < 0.8        -> CERCA           -> ALEJAR
# 0.8 <= depth <= 1  -> BIEN_DISTANCIA  -> PARAR
# depth > 1.0        -> LEJOS           -> ACERCAR

MIN_OK_DISTANCE = 0.8
MAX_OK_DISTANCE = 1.0

NEAR_THRESHOLD = MIN_OK_DISTANCE
FAR_THRESHOLD = MAX_OK_DISTANCE


# ============================================================
# ESTABILIZACIÓN DE DISTANCIA
# ============================================================

# Histéresis asimétrica:
# - CERCA se activa estrictamente por debajo de MIN_OK_DISTANCE.
# - LEJOS tiene margen cuando ya estaba en BIEN_DISTANCIA.
DISTANCE_HYSTERESIS_NEAR = 0.00
DISTANCE_HYSTERESIS_FAR = 0.12


# ============================================================
# BLOQUEO DE POSICIÓN ALCANZADA
# ============================================================

POSITION_LOCK_ENABLED = True

# Una vez el robot está colocado, solo sale del bloqueo
# si la profundidad cambia de forma clara.
POSITION_LOCK_NEAR_EXIT = 0.88
POSITION_LOCK_FAR_EXIT = 1.15

# También sale del bloqueo si el target se descentra mucho.
POSITION_LOCK_CENTER_EXIT_PX = 90.0

# Número de decisiones consecutivas fuera de rango para desbloquear.
POSITION_LOCK_CONFIRM_FRAMES = 3

position_lock = {
    "active": False,
    "exit_counter": 0,
    "target": None,
}


# ============================================================
# LÓGICA DE CENTRADO HORIZONTAL
# ============================================================

IMAGE_CENTER_X = COLOR_WIDTH / 2.0

# Margen respecto al centro de la imagen.
# Si el target está dentro de ±50 px, se considera centrado.
CENTER_TOLERANCE_PX = 50.0


# ============================================================
# VISUALIZACIÓN
# ============================================================

DRAW_DISTANCE_OVERLAY = True
DRAW_DEPTH_TARGET = False
PRINT_DEBUG_EVERY_N_FRAMES = 30


# ============================================================
# ENGINE RTMPOSE
# ============================================================

engine = LandmarksEngine(
    device=DEVICE,
    thr=BODY_THR,
    face_thr=FACE_THR,
)


depth_buffers = {
    "thyroid": deque(maxlen=SMOOTH_LEN),
    "prostate": deque(maxlen=SMOOTH_LEN),
}

state_buffers = {
    "thyroid": deque(maxlen=STATE_LEN),
    "prostate": deque(maxlen=STATE_LEN),
}

last_depth_state = {
    "thyroid": "SIN_MEDIDA",
    "prostate": "SIN_MEDIDA",
}

da3_cache = {
    "thyroid": None,
    "prostate": None,
}

da3_errors_cache = {
    "thyroid": None,
    "prostate": None,
}

da3_fresh_cache = {
    "thyroid": False,
    "prostate": False,
}

last_da3_frame_updated = {
    "frame": -1,
}


# ============================================================
# CLIENTE DEPTH ANYTHING 3
# ============================================================

class DepthAnythingClient:
    def __init__(self, url, timeout=60):
        self.url = url
        self.timeout = timeout

    def get_depths_at_points(self, frame_bgr, points, window=25):
        """
        Envía una sola imagen y varios puntos al servidor DA3.
        """

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


da3_client = DepthAnythingClient(DA3_SERVER_URL)


# ============================================================
# UTILIDADES
# ============================================================

def point_or_none(p):
    if p is None:
        return None

    return {
        "x": float(p[0]),
        "y": float(p[1]),
    }


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
    """
    Recoge thyroid y prostate si están disponibles y dentro de la imagen.
    """

    points = {}

    if landmarks is None:
        return points

    for target_name in DEPTH_TARGETS:
        point = landmarks.get(target_name)
        xy = point_to_xy(point)

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


# ============================================================
# CLASIFICACIÓN DE DISTANCIA
# ============================================================

def classify_depth(depth_value):
    """
    depth < 0.8        -> CERCA
    0.8 <= depth <= 1  -> BIEN_DISTANCIA
    depth > 1.0        -> LEJOS
    """

    if depth_value is None:
        return "SIN_MEDIDA"

    if depth_value < MIN_OK_DISTANCE:
        return "CERCA"

    if depth_value > MAX_OK_DISTANCE:
        return "LEJOS"

    return "BIEN_DISTANCIA"


def classify_depth_with_hysteresis(depth_value, previous_state):
    """
    Clasifica la profundidad con histéresis asimétrica.

    - CERCA se activa siempre por debajo de MIN_OK_DISTANCE.
    - LEJOS, cuando ya estaba en buena distancia, necesita superar
      MAX_OK_DISTANCE + DISTANCE_HYSTERESIS_FAR.
    """

    if depth_value is None:
        return "SIN_MEDIDA"

    if previous_state == "BIEN_DISTANCIA":
        near_limit = MIN_OK_DISTANCE
        far_limit = MAX_OK_DISTANCE + DISTANCE_HYSTERESIS_FAR

        if depth_value < near_limit:
            return "CERCA"

        if depth_value > far_limit:
            return "LEJOS"

        return "BIEN_DISTANCIA"

    return classify_depth(depth_value)


def action_from_state(state):
    if state == "CERCA":
        return "ALEJAR"

    if state == "BIEN_DISTANCIA":
        return "PARAR"

    if state == "LEJOS":
        return "ACERCAR"

    return "SIN_ACCION"


# ============================================================
# CLASIFICACIÓN DE CENTRADO
# ============================================================

def classify_horizontal_position(point):
    """
    Determina si el target está a la izquierda, centrado o a la derecha
    del centro de la imagen.
    """

    if point is None:
        return "SIN_MEDIDA", None

    try:
        x = float(point["x"])
    except Exception:
        return "SIN_MEDIDA", None

    error_x = x - IMAGE_CENTER_X

    if error_x < -CENTER_TOLERANCE_PX:
        return "IZQUIERDA", float(error_x)

    if error_x > CENTER_TOLERANCE_PX:
        return "DERECHA", float(error_x)

    return "CENTRADO", float(error_x)


def centering_action_from_state(horizontal_state):
    """
    Acción de corrección horizontal.
    """

    if horizontal_state == "IZQUIERDA":
        return "IZQUIERDA"

    if horizontal_state == "DERECHA":
        return "DERECHA"

    if horizontal_state == "CENTRADO":
        return "CENTRADO"

    return "SIN_ACCION"


def combine_actions(distance_action, centering_action):
    """
    Combina acción de distancia + acción de centrado.

    Casos principales:
    - Si no hay medición válida -> SIN_ACCION.
    - Si está bien de distancia y centrado -> PARAR.
    - Si está bien de distancia pero descentrado -> IZQUIERDA / DERECHA.
    - Si está lejos/cerca y descentrado -> acción combinada.
    """

    if distance_action is None:
        distance_action = "SIN_ACCION"

    if centering_action is None:
        centering_action = "SIN_ACCION"

    if distance_action == "SIN_ACCION":
        return "SIN_ACCION"

    if distance_action == "PARAR":
        if centering_action == "IZQUIERDA":
            return "IZQUIERDA"

        if centering_action == "DERECHA":
            return "DERECHA"

        return "PARAR"

    if centering_action in ["SIN_ACCION", "CENTRADO"]:
        return distance_action

    return f"{distance_action}_{centering_action}"


def majority_state(buffer):
    valid = [s for s in buffer if s != "SIN_MEDIDA"]

    if not valid:
        return "SIN_MEDIDA"

    return Counter(valid).most_common(1)[0][0]


# ============================================================
# PAYLOADS POR DEFECTO
# ============================================================

def default_da3_compare_payload():
    return {
        "enabled": True,
        "da3_depth": None,
        "abs_diff": None,
        "rel_diff": None,
        "agreement_ok": True,
        "warning": "",
        "is_fresh": False,
        "mode": "da3_only_batch",
    }


def default_target_payload():
    return {
        "available": False,
        "point": None,

        "raw_depth": None,
        "smooth_depth": None,

        "instant_state": "SIN_MEDIDA",
        "stable_state": "SIN_MEDIDA",

        "action": "SIN_ACCION",

        "distance_action": "SIN_ACCION",
        "horizontal_state": "SIN_MEDIDA",
        "centering_action": "SIN_ACCION",
        "horizontal_error_px": None,

        "da3_depth": None,
        "da3_compare": default_da3_compare_payload(),
        "error": None,
    }


def default_distance_payload():
    return {
        "enabled": True,
        "source": "depth_anything3_batch",

        "target": None,
        "target_point": None,
        "measurement_allowed": False,

        "position_lock_active": False,
        "position_lock_reason": "",
        "position_lock_near_exit": POSITION_LOCK_NEAR_EXIT,
        "position_lock_far_exit": POSITION_LOCK_FAR_EXIT,
        "position_lock_center_exit_px": POSITION_LOCK_CENTER_EXIT_PX,
        "position_lock_confirm_frames": POSITION_LOCK_CONFIRM_FRAMES,

        "raw_depth": None,
        "smooth_depth": None,

        "instant_state": "SIN_MEDIDA",
        "stable_state": "SIN_MEDIDA",

        "action": "SIN_ACCION",

        "distance_action": "SIN_ACCION",
        "horizontal_state": "SIN_MEDIDA",
        "centering_action": "SIN_ACCION",
        "horizontal_error_px": None,
        "center_tolerance_px": CENTER_TOLERANCE_PX,

        "da3_depth": None,
        "da3_compare": default_da3_compare_payload(),

        "near_threshold": NEAR_THRESHOLD,
        "far_threshold": FAR_THRESHOLD,
        "min_ok_distance": MIN_OK_DISTANCE,
        "max_ok_distance": MAX_OK_DISTANCE,
        "distance_hysteresis_near": DISTANCE_HYSTERESIS_NEAR,
        "distance_hysteresis_far": DISTANCE_HYSTERESIS_FAR,
        "window": DEPTH_WINDOW,

        "targets": {
            "thyroid": default_target_payload(),
            "prostate": default_target_payload(),
        },

        "error": None,
    }


# ============================================================
# PROFUNDIDAD DA3
# ============================================================

def reset_target_cache(target_name, error_msg=None):
    da3_cache[target_name] = None
    da3_errors_cache[target_name] = error_msg
    da3_fresh_cache[target_name] = False

    depth_buffers[target_name].clear()
    state_buffers[target_name].append("SIN_MEDIDA")
    last_depth_state[target_name] = "SIN_MEDIDA"


def update_da3_depths(frame_bgr, landmarks, frame_count, measurement_allowed):
    """
    Hace UNA sola llamada a DA3 para todos los puntos disponibles.
    """

    for name in DEPTH_TARGETS:
        da3_fresh_cache[name] = False

    if not measurement_allowed:
        return False

    if last_da3_frame_updated["frame"] == frame_count:
        return False

    should_query = (
        frame_count % DA3_EVERY_N_FRAMES == 0
        or all(da3_cache[name] is None for name in DEPTH_TARGETS)
    )

    if not should_query:
        return False

    last_da3_frame_updated["frame"] = frame_count

    points = collect_available_points(landmarks)

    if not points:
        for name in DEPTH_TARGETS:
            reset_target_cache(name, "no_valid_point")
        return False

    try:
        depths, errors = da3_client.get_depths_at_points(
            frame_bgr=frame_bgr,
            points=points,
            window=DEPTH_WINDOW,
        )

        for name in DEPTH_TARGETS:
            if name not in points:
                reset_target_cache(name, "point_outside_image_or_missing")
                continue

            if name in depths:
                value = depths.get(name, None)

                if value is not None:
                    da3_cache[name] = float(value)
                else:
                    da3_cache[name] = None

                if isinstance(errors, dict):
                    da3_errors_cache[name] = errors.get(name, None)
                else:
                    da3_errors_cache[name] = None

                da3_fresh_cache[name] = True
            else:
                da3_cache[name] = None
                da3_errors_cache[name] = "not_returned"
                da3_fresh_cache[name] = True

        return True

    except Exception as e:
        for name in DEPTH_TARGETS:
            da3_errors_cache[name] = str(e)
            da3_fresh_cache[name] = True

        return False


def compute_single_target_depth(landmarks, target_name, measurement_allowed):
    result = default_target_payload()

    if landmarks is None:
        return result

    point = landmarks.get(target_name)
    xy = point_to_xy(point)

    if xy is None:
        reset_target_cache(target_name, "missing_landmark")
        return result

    x, y = xy

    result["available"] = True
    result["point"] = {
        "x": float(x),
        "y": float(y),
    }

    horizontal_state, horizontal_error_px = classify_horizontal_position(result["point"])
    centering_action = centering_action_from_state(horizontal_state)

    result["horizontal_state"] = horizontal_state
    result["horizontal_error_px"] = horizontal_error_px
    result["centering_action"] = centering_action

    if not point_inside_image(x, y):
        reset_target_cache(target_name, "point_outside_image")
        result["error"] = "point_outside_image"
        return result

    if not measurement_allowed:
        state_buffers[target_name].append("SIN_MEDIDA")

        if len(state_buffers[target_name]) == state_buffers[target_name].maxlen:
            if all(s == "SIN_MEDIDA" for s in state_buffers[target_name]):
                depth_buffers[target_name].clear()
                last_depth_state[target_name] = "SIN_MEDIDA"

        return result

    da3_depth = da3_cache.get(target_name, None)
    da3_error = da3_errors_cache.get(target_name, None)
    is_fresh = da3_fresh_cache.get(target_name, False)

    smooth_depth = None

    # Solo añadimos una medida nueva cuando DA3 acaba de calcularla.
    # Así evitamos llenar el buffer con valores repetidos de la caché.
    if da3_depth is not None and is_fresh:
        depth_buffers[target_name].append(da3_depth)

    if len(depth_buffers[target_name]) > 0:
        # Mediana en vez de media: más robusta ante saltos puntuales.
        smooth_depth = float(np.median(depth_buffers[target_name]))

    previous_state = last_depth_state.get(target_name, "SIN_MEDIDA")

    instant_state = classify_depth_with_hysteresis(
        depth_value=smooth_depth,
        previous_state=previous_state,
    )

    state_buffers[target_name].append(instant_state)

    stable_state = majority_state(state_buffers[target_name])

    if stable_state != "SIN_MEDIDA":
        last_depth_state[target_name] = stable_state

    distance_action = action_from_state(stable_state)
    action = combine_actions(distance_action, centering_action)

    da3_compare = default_da3_compare_payload()
    da3_compare["da3_depth"] = float(da3_depth) if da3_depth is not None else None
    da3_compare["is_fresh"] = bool(is_fresh)
    da3_compare["agreement_ok"] = da3_error is None
    da3_compare["warning"] = da3_error if da3_error is not None else ""

    result.update({
        "raw_depth": float(da3_depth) if da3_depth is not None else None,
        "smooth_depth": float(smooth_depth) if smooth_depth is not None else None,
        "instant_state": instant_state,
        "stable_state": stable_state,

        "action": action,
        "distance_action": distance_action,
        "horizontal_state": horizontal_state,
        "centering_action": centering_action,
        "horizontal_error_px": horizontal_error_px,

        "da3_depth": float(da3_depth) if da3_depth is not None else None,
        "da3_compare": da3_compare,
        "error": da3_error,
    })

    return result


# ============================================================
# BLOQUEO DE POSICIÓN
# ============================================================

def apply_position_lock(distance):
    """
    Bloquea la acción en PARAR cuando el robot ya está centrado
    y a buena distancia.

    Evita oscilaciones causadas por pequeñas variaciones de DA3.
    """

    if not POSITION_LOCK_ENABLED:
        distance["position_lock_active"] = False
        distance["position_lock_reason"] = "disabled"
        return distance

    target = distance.get("target", None)
    smooth_depth = distance.get("smooth_depth", None)
    stable_state = distance.get("stable_state", "SIN_MEDIDA")
    horizontal_state = distance.get("horizontal_state", "SIN_MEDIDA")
    horizontal_error_px = distance.get("horizontal_error_px", None)

    # Si no hay medida, por seguridad mantenemos PARAR si ya estaba bloqueado.
    if target is None or smooth_depth is None:
        if position_lock["active"]:
            distance["action"] = "PARAR"
            distance["distance_action"] = "PARAR"
            distance["position_lock_active"] = True
            distance["position_lock_reason"] = "locked_without_measurement"
        else:
            distance["position_lock_active"] = False
            distance["position_lock_reason"] = "no_measurement"

        return distance

    # ========================================================
    # ENTRAR EN BLOQUEO
    # ========================================================

    if not position_lock["active"]:
        if stable_state == "BIEN_DISTANCIA" and horizontal_state == "CENTRADO":
            position_lock["active"] = True
            position_lock["exit_counter"] = 0
            position_lock["target"] = target

            distance["action"] = "PARAR"
            distance["distance_action"] = "PARAR"
            distance["position_lock_active"] = True
            distance["position_lock_reason"] = "entered_lock"

        else:
            distance["position_lock_active"] = False
            distance["position_lock_reason"] = "not_locked"

        return distance

    # ========================================================
    # YA ESTÁ BLOQUEADO
    # ========================================================

    too_close = smooth_depth < POSITION_LOCK_NEAR_EXIT
    too_far = smooth_depth > POSITION_LOCK_FAR_EXIT

    too_uncentered = False

    if horizontal_error_px is not None:
        too_uncentered = abs(float(horizontal_error_px)) > POSITION_LOCK_CENTER_EXIT_PX

    should_exit_lock = too_close or too_far or too_uncentered

    if should_exit_lock:
        position_lock["exit_counter"] += 1
    else:
        position_lock["exit_counter"] = 0

    # Mientras no haya suficientes frames confirmando el cambio,
    # mantenemos PARAR.
    if position_lock["exit_counter"] < POSITION_LOCK_CONFIRM_FRAMES:
        distance["action"] = "PARAR"
        distance["distance_action"] = "PARAR"
        distance["position_lock_active"] = True
        distance["position_lock_reason"] = "holding_lock"
        return distance

    # Salir del bloqueo
    position_lock["active"] = False
    position_lock["exit_counter"] = 0
    position_lock["target"] = None

    distance["position_lock_active"] = False

    if too_close:
        distance["position_lock_reason"] = "exit_too_close"
    elif too_far:
        distance["position_lock_reason"] = "exit_too_far"
    elif too_uncentered:
        distance["position_lock_reason"] = "exit_uncentered"
    else:
        distance["position_lock_reason"] = "exit_unknown"

    return distance


def compute_distance_from_da3(frame_bgr, landmarks, orientation, frame_count):
    distance = default_distance_payload()

    available_points = collect_available_points(landmarks)

    measurement_allowed = (
        orientation == "front"
        and landmarks is not None
        and len(available_points) > 0
    )

    distance["measurement_allowed"] = bool(measurement_allowed)

    update_da3_depths(
        frame_bgr=frame_bgr,
        landmarks=landmarks,
        frame_count=frame_count,
        measurement_allowed=measurement_allowed,
    )

    for target_name in DEPTH_TARGETS:
        target_result = compute_single_target_depth(
            landmarks=landmarks,
            target_name=target_name,
            measurement_allowed=measurement_allowed,
        )

        distance["targets"][target_name] = target_result

    selected_name = None
    selected_data = None

    for name in PRIMARY_TARGET_PRIORITY:
        data = distance["targets"].get(name)

        if data is None:
            continue

        if data.get("available", False) and data.get("smooth_depth") is not None:
            selected_name = name
            selected_data = data
            break

    if selected_data is not None:
        distance["target"] = selected_name
        distance["target_point"] = selected_data.get("point")
        distance["raw_depth"] = selected_data.get("raw_depth")
        distance["smooth_depth"] = selected_data.get("smooth_depth")
        distance["instant_state"] = selected_data.get("instant_state", "SIN_MEDIDA")
        distance["stable_state"] = selected_data.get("stable_state", "SIN_MEDIDA")

        distance["action"] = selected_data.get("action", "SIN_ACCION")
        distance["distance_action"] = selected_data.get("distance_action", "SIN_ACCION")
        distance["horizontal_state"] = selected_data.get("horizontal_state", "SIN_MEDIDA")
        distance["centering_action"] = selected_data.get("centering_action", "SIN_ACCION")
        distance["horizontal_error_px"] = selected_data.get("horizontal_error_px", None)

        distance["da3_depth"] = selected_data.get("da3_depth")
        distance["da3_compare"] = selected_data.get(
            "da3_compare",
            default_da3_compare_payload(),
        )
    else:
        distance["target"] = None
        distance["target_point"] = None
        distance["raw_depth"] = None
        distance["smooth_depth"] = None
        distance["instant_state"] = "SIN_MEDIDA"
        distance["stable_state"] = "SIN_MEDIDA"

        distance["action"] = "SIN_ACCION"
        distance["distance_action"] = "SIN_ACCION"
        distance["horizontal_state"] = "SIN_MEDIDA"
        distance["centering_action"] = "SIN_ACCION"
        distance["horizontal_error_px"] = None

        distance["da3_depth"] = None
        distance["da3_compare"] = default_da3_compare_payload()

    distance = apply_position_lock(distance)

    return distance


# ============================================================
# DIBUJO
# ============================================================

def draw_text(img, text, pos, color=(255, 255, 255), scale=0.5, thickness=1):
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


def draw_distance_overlay(img, orientation, distance):
    if not DRAW_DISTANCE_OVERLAY:
        return img

    out = img.copy()

    selected_target = distance.get("target", None)

    stable_state = distance.get("stable_state", "SIN_MEDIDA")
    horizontal_state = distance.get("horizontal_state", "SIN_MEDIDA")
    final_action = distance.get("action", "SIN_ACCION")

    smooth_depth = distance.get("smooth_depth", None)

    if stable_state == "CERCA":
        state_color = (0, 0, 255)
    elif stable_state == "BIEN_DISTANCIA":
        state_color = (0, 255, 0)
    elif stable_state == "LEJOS":
        state_color = (0, 165, 255)
    else:
        state_color = (180, 180, 180)

    if orientation == "front":
        orientation_color = (0, 255, 0)
    else:
        orientation_color = (0, 0, 255)

    draw_text(
        out,
        f"orient: {orientation}",
        (20, 32),
        orientation_color,
        scale=0.48
    )

    draw_text(
        out,
        f"target: {selected_target}",
        (20, 54),
        state_color,
        scale=0.48
    )

    if smooth_depth is not None:
        draw_text(
            out,
            f"DA3: {smooth_depth:.3f}",
            (20, 76),
            state_color,
            scale=0.48
        )
    else:
        draw_text(
            out,
            "DA3: None",
            (20, 76),
            state_color,
            scale=0.48
        )

    draw_text(
        out,
        f"state: {stable_state} / {horizontal_state}",
        (20, 98),
        state_color,
        scale=0.48
    )

    draw_text(
        out,
        f"action: {final_action}",
        (20, 120),
        (0, 165, 255),
        scale=0.48
    )

    return out


# ============================================================
# GUARDADO
# ============================================================

def save_landmarks_json(
    landmarks,
    keypoints,
    scores,
    orientation,
    orientation_debug=None,
    distance=None,
):
    payload = {
        "valid": landmarks is not None,
        "source": "realsense_rgb_da3_batch",
        "orientation": orientation,
        "measurement_allowed": False,
        "neck_base": None,
        "pelvis": None,
        "thyroid": None,
        "prostate": None,
        "keypoints": [],
        "orientation_debug": orientation_debug if orientation_debug is not None else {},
        "distance": distance if distance is not None else default_distance_payload(),
    }

    if landmarks is not None:
        if distance is not None and isinstance(distance, dict):
            payload["measurement_allowed"] = bool(
                distance.get("measurement_allowed", False)
            )
        else:
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
                "score": float(sc),
            })

    tmp_path = str(LATEST_LANDMARKS_PATH) + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    os.replace(tmp_path, LATEST_LANDMARKS_PATH)


def save_frame_atomic(path, img):
    tmp_path = str(path) + ".tmp.jpg"
    cv2.imwrite(tmp_path, img)
    os.replace(tmp_path, path)


# ============================================================
# MAIN
# ============================================================

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        COLOR_WIDTH,
        COLOR_HEIGHT,
        rs.format.bgr8,
        FPS,
    )

    print("Iniciando RealSense como cámara RGB...")

    pipeline.start(config)

    print("RealSense iniciada.")
    print("Modo: RealSense RGB + Depth Anything 3 batch.")
    print("No se usa profundidad de RealSense.")
    print(f"Servidor DA3: {DA3_SERVER_URL}")
    print(f"DA3 cada {DA3_EVERY_N_FRAMES} frame(s).")
    print("Se pide thyroid + prostate en una sola llamada.")
    print("Acciones: distancia + centrado horizontal.")
    print("Rango DA3:")
    print(f"  depth < {MIN_OK_DISTANCE} -> CERCA -> ALEJAR")
    print(f"  {MIN_OK_DISTANCE} <= depth <= {MAX_OK_DISTANCE} -> BIEN_DISTANCIA -> PARAR")
    print(f"  depth > {MAX_OK_DISTANCE} -> LEJOS -> ACERCAR")
    print("Histéresis cuando ya está en BIEN_DISTANCIA:")
    print(f"  depth < {MIN_OK_DISTANCE - DISTANCE_HYSTERESIS_NEAR:.2f} -> CERCA -> ALEJAR")
    print(f"  depth > {MAX_OK_DISTANCE + DISTANCE_HYSTERESIS_FAR:.2f} -> LEJOS -> ACERCAR")
    print("Bloqueo de posición:")
    print(f"  active={POSITION_LOCK_ENABLED}")
    print(f"  near_exit={POSITION_LOCK_NEAR_EXIT}")
    print(f"  far_exit={POSITION_LOCK_FAR_EXIT}")
    print(f"  center_exit_px={POSITION_LOCK_CENTER_EXIT_PX}")
    print(f"  confirm_frames={POSITION_LOCK_CONFIRM_FRAMES}")
    print("Procesando RTMPose + landmarks + distancia DA3...")
    print("Pulsa ESC en la ventana o Ctrl+C en terminal para salir.")

    frame_count = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            frame_bgr = np.asanyarray(color_frame.get_data())

            result = engine.process_frame(frame_bgr)

            final_img = result["image"]
            landmarks = result["landmarks"]
            keypoints = result["keypoints"]
            scores = result["scores"]
            orientation = result.get("orientation", "not_front")
            orientation_debug = result.get("orientation_debug", {})

            distance = compute_distance_from_da3(
                frame_bgr=frame_bgr,
                landmarks=landmarks,
                orientation=orientation,
                frame_count=frame_count,
            )

            final_img = draw_distance_overlay(
                final_img,
                orientation=orientation,
                distance=distance,
            )

            save_frame_atomic(LATEST_FRAME_PATH, final_img)

            save_landmarks_json(
                landmarks=landmarks,
                keypoints=keypoints,
                scores=scores,
                orientation=orientation,
                orientation_debug=orientation_debug,
                distance=distance,
            )

            frame_count += 1

            if frame_count % PRINT_DEBUG_EVERY_N_FRAMES == 0:
                targets_debug = distance.get("targets", {})
                thyroid_debug = targets_debug.get("thyroid", {})
                prostate_debug = targets_debug.get("prostate", {})

                print("======================================")
                print(f"RGB shape: {frame_bgr.shape}")
                print(f"Orientation: {orientation}")

                print("--- Selected depth DA3 ---")
                print(f"Target: {distance.get('target')}")
                print(f"Target point: {distance.get('target_point')}")
                print(f"DA3 raw depth: {distance.get('da3_depth')}")
                print(f"Smooth depth: {distance.get('smooth_depth')}")
                print(f"Distance state: {distance.get('stable_state')}")
                print(f"Distance action: {distance.get('distance_action')}")
                print(f"Horizontal state: {distance.get('horizontal_state')}")
                print(f"Centering action: {distance.get('centering_action')}")
                print(f"Horizontal error px: {distance.get('horizontal_error_px')}")
                print(f"Combined action: {distance.get('action')}")
                print(f"Measurement allowed: {distance.get('measurement_allowed')}")
                print(f"Position lock active: {distance.get('position_lock_active')}")
                print(f"Position lock reason: {distance.get('position_lock_reason')}")
                print(f"Error: {distance.get('error')}")

                print("--- Thyroid DA3 ---")
                print(f"available: {thyroid_debug.get('available')}")
                print(f"point: {thyroid_debug.get('point')}")
                print(f"da3: {thyroid_debug.get('da3_depth')}")
                print(f"smooth: {thyroid_debug.get('smooth_depth')}")
                print(f"state: {thyroid_debug.get('stable_state')}")
                print(f"distance_action: {thyroid_debug.get('distance_action')}")
                print(f"horizontal_state: {thyroid_debug.get('horizontal_state')}")
                print(f"centering_action: {thyroid_debug.get('centering_action')}")
                print(f"horizontal_error_px: {thyroid_debug.get('horizontal_error_px')}")
                print(f"action: {thyroid_debug.get('action')}")
                print(f"error: {thyroid_debug.get('error')}")

                print("--- Prostate DA3 ---")
                print(f"available: {prostate_debug.get('available')}")
                print(f"point: {prostate_debug.get('point')}")
                print(f"da3: {prostate_debug.get('da3_depth')}")
                print(f"smooth: {prostate_debug.get('smooth_depth')}")
                print(f"state: {prostate_debug.get('stable_state')}")
                print(f"distance_action: {prostate_debug.get('distance_action')}")
                print(f"horizontal_state: {prostate_debug.get('horizontal_state')}")
                print(f"centering_action: {prostate_debug.get('centering_action')}")
                print(f"horizontal_error_px: {prostate_debug.get('horizontal_error_px')}")
                print(f"action: {prostate_debug.get('action')}")
                print(f"error: {prostate_debug.get('error')}")

            cv2.imshow("RealSense RTMPose RGB + DA3 Batch", final_img)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

    except KeyboardInterrupt:
        pass

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense detenida.")


if __name__ == "__main__":
    main()
