import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer


COCO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0]


def point_between(p_from, p_to, alpha):
    return [
        p_from[0] + alpha * (p_to[0] - p_from[0]),
        p_from[1] + alpha * (p_to[1] - p_from[1]),
    ]


def is_valid(conf, threshold=0.3):
    return conf is not None and float(conf) > threshold


def ensure_bgr(img):
    if img is None:
        return None
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def pick_best_person(people):
    if not people:
        return None
    return max(
        people,
        key=lambda p: float(np.mean(np.array(p.get("keypoint_scores", [0]), dtype=float)))
    )


def estimate_orientation(
    keypoints,
    scores,
    body_threshold=0.3,
    face_threshold=0.5
):
    """
    Decide front / not_front using facial + shoulder geometry.
    We do NOT trust face keypoints only by presence.
    We require a plausible frontal configuration.
    """
    nose = keypoints[COCO_KEYPOINTS["nose"]]
    left_eye = keypoints[COCO_KEYPOINTS["left_eye"]]
    right_eye = keypoints[COCO_KEYPOINTS["right_eye"]]
    left_sh = keypoints[COCO_KEYPOINTS["left_shoulder"]]
    right_sh = keypoints[COCO_KEYPOINTS["right_shoulder"]]

    nose_conf = scores[COCO_KEYPOINTS["nose"]]
    left_eye_conf = scores[COCO_KEYPOINTS["left_eye"]]
    right_eye_conf = scores[COCO_KEYPOINTS["right_eye"]]
    left_sh_conf = scores[COCO_KEYPOINTS["left_shoulder"]]
    right_sh_conf = scores[COCO_KEYPOINTS["right_shoulder"]]

    face_ok = (
        is_valid(nose_conf, face_threshold)
        and is_valid(left_eye_conf, face_threshold)
        and is_valid(right_eye_conf, face_threshold)
    )

    shoulders_ok = (
        is_valid(left_sh_conf, body_threshold)
        and is_valid(right_sh_conf, body_threshold)
    )

    debug = {
        "face_ok": bool(face_ok),
        "shoulders_ok": bool(shoulders_ok),
        "nose_conf": float(nose_conf) if nose_conf is not None else None,
        "left_eye_conf": float(left_eye_conf) if left_eye_conf is not None else None,
        "right_eye_conf": float(right_eye_conf) if right_eye_conf is not None else None,
        "left_shoulder_conf": float(left_sh_conf) if left_sh_conf is not None else None,
        "right_shoulder_conf": float(right_sh_conf) if right_sh_conf is not None else None,
    }

    if not face_ok or not shoulders_ok:
        debug.update({
            "orientation_raw": "not_front",
            "front_score": 0,
            "reason": "missing_face_or_shoulders"
        })
        return "not_front", False, debug

    shoulder_mid = midpoint(left_sh, right_sh)
    shoulder_width = abs(right_sh[0] - left_sh[0])
    eye_dist = abs(right_eye[0] - left_eye[0])

    debug["shoulder_width"] = float(shoulder_width)
    debug["eye_distance"] = float(eye_dist)

    if shoulder_width < 10 or eye_dist < 3:
        debug.update({
            "orientation_raw": "not_front",
            "front_score": 0,
            "reason": "too_small_geometry"
        })
        return "not_front", False, debug

    nose_between_eyes = min(left_eye[0], right_eye[0]) <= nose[0] <= max(left_eye[0], right_eye[0])

    eye_y_diff = abs(left_eye[1] - right_eye[1])
    eyes_level = eye_y_diff < 0.08 * shoulder_width

    nose_center_offset = abs(nose[0] - shoulder_mid[0])
    nose_centered = nose_center_offset < 0.18 * shoulder_width

    shoulder_y = (left_sh[1] + right_sh[1]) / 2.0
    eyes_above_shoulders = left_eye[1] < shoulder_y and right_eye[1] < shoulder_y

    face_ratio = eye_dist / shoulder_width
    face_ratio_ok = 0.08 < face_ratio < 0.55

    dist_nose_left = abs(nose[0] - left_eye[0])
    dist_nose_right = abs(right_eye[0] - nose[0])
    symmetry_ok = abs(dist_nose_left - dist_nose_right) < 0.12 * shoulder_width

    checks = {
        "nose_between_eyes": bool(nose_between_eyes),
        "eyes_level": bool(eyes_level),
        "nose_centered": bool(nose_centered),
        "eyes_above_shoulders": bool(eyes_above_shoulders),
        "face_ratio_ok": bool(face_ratio_ok),
        "symmetry_ok": bool(symmetry_ok),
    }

    front_score = sum(checks.values())
    is_front = front_score >= 5

    debug.update(checks)
    debug.update({
        "eye_y_diff": float(eye_y_diff),
        "nose_center_offset": float(nose_center_offset),
        "face_ratio": float(face_ratio),
        "dist_nose_left": float(dist_nose_left),
        "dist_nose_right": float(dist_nose_right),
        "front_score": int(front_score),
        "orientation_raw": "front" if is_front else "not_front",
        "reason": "geometry_check"
    })

    return ("front" if is_front else "not_front"), is_front, debug


def compute_landmarks(
    keypoints,
    scores,
    threshold=0.3,
    thyroid_alpha=0.25,
    prostate_offset_alpha=0.15,
    measurement_allowed=False
):
    l_sh = keypoints[COCO_KEYPOINTS["left_shoulder"]]
    r_sh = keypoints[COCO_KEYPOINTS["right_shoulder"]]
    l_hip = keypoints[COCO_KEYPOINTS["left_hip"]]
    r_hip = keypoints[COCO_KEYPOINTS["right_hip"]]
    nose = keypoints[COCO_KEYPOINTS["nose"]]

    l_sh_conf = scores[COCO_KEYPOINTS["left_shoulder"]]
    r_sh_conf = scores[COCO_KEYPOINTS["right_shoulder"]]
    l_hip_conf = scores[COCO_KEYPOINTS["left_hip"]]
    r_hip_conf = scores[COCO_KEYPOINTS["right_hip"]]
    nose_conf = scores[COCO_KEYPOINTS["nose"]]

    neck_base = None
    thyroid = None
    pelvis = None
    prostate = None

    if is_valid(l_sh_conf, threshold) and is_valid(r_sh_conf, threshold):
        neck_base = midpoint(l_sh, r_sh)

    if is_valid(l_hip_conf, threshold) and is_valid(r_hip_conf, threshold):
        pelvis = midpoint(l_hip, r_hip)

    if neck_base is not None and is_valid(nose_conf, threshold):
        thyroid = point_between(neck_base, nose, thyroid_alpha)

    if pelvis is not None:
        hip_width = abs(r_hip[0] - l_hip[0])
        prostate = [pelvis[0], pelvis[1] + prostate_offset_alpha * hip_width]

    return {
        "neck_base": neck_base,
        "thyroid": thyroid,
        "pelvis": pelvis,
        "prostate": prostate,
        "measurement_allowed": measurement_allowed,
    }


def draw_interest_points(img, landmarks, orientation="not_front", orientation_debug=None):
    """
    Dibuja únicamente los puntos anatómicos de interés.

    Se mantiene:
    - thyroid
    - prostate
    - línea pelvis-prostate

    Se elimina de la pantalla:
    - orientation: front / not_front
    - front_score
    """
    out = img.copy()

    thyroid = landmarks["thyroid"]
    prostate = landmarks["prostate"]
    pelvis = landmarks["pelvis"]

    if thyroid is not None:
        cv2.circle(out, (int(thyroid[0]), int(thyroid[1])), 7, (0, 165, 255), -1)
        cv2.putText(
            out,
            "thyroid",
            (int(thyroid[0]) + 8, int(thyroid[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
        )

    if prostate is not None:
        cv2.circle(out, (int(prostate[0]), int(prostate[1])), 7, (255, 0, 255), -1)
        cv2.putText(
            out,
            "prostate",
            (int(prostate[0]) + 8, int(prostate[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
        )

    if pelvis is not None and prostate is not None:
        cv2.line(
            out,
            (int(pelvis[0]), int(pelvis[1])),
            (int(prostate[0]), int(prostate[1])),
            (255, 0, 255),
            2,
        )

    return out


class LandmarksEngine:
    def __init__(
        self,
        device="cuda:0",
        thr=0.3,
        face_thr=0.5,
        thyroid_alpha=0.25,
        prostate_offset_alpha=0.15,
        front_frames_required=3,
        not_front_frames_required=2
    ):
        self.device = device
        self.thr = thr
        self.face_thr = face_thr
        self.thyroid_alpha = thyroid_alpha
        self.prostate_offset_alpha = prostate_offset_alpha
        self.inferencer = MMPoseInferencer(pose2d="human", device=device)

        self.front_frames_required = front_frames_required
        self.not_front_frames_required = not_front_frames_required
        self.front_counter = 0
        self.not_front_counter = 0
        self.current_orientation = "not_front"

    def _smooth_orientation(self, raw_orientation):
        if raw_orientation == "front":
            self.front_counter += 1
            self.not_front_counter = 0
        else:
            self.not_front_counter += 1
            self.front_counter = 0

        if self.front_counter >= self.front_frames_required:
            self.current_orientation = "front"
        elif self.not_front_counter >= self.not_front_frames_required:
            self.current_orientation = "not_front"

        return self.current_orientation

    def process_frame(self, frame_bgr):
        result = next(self.inferencer(frame_bgr, return_vis=True))

        predictions = result.get("predictions", None)
        visualizations = result.get("visualization", None)

        if not predictions or not predictions[0]:
            return {
                "success": False,
                "image": frame_bgr.copy(),
                "landmarks": None,
                "keypoints": None,
                "scores": None,
                "orientation": "not_front",
                "orientation_debug": {
                    "reason": "no_predictions",
                    "orientation_raw": "not_front",
                    "orientation_smoothed": "not_front",
                    "front_score": 0,
                },
            }

        people = predictions[0]
        best_person = pick_best_person(people)

        if best_person is None:
            return {
                "success": False,
                "image": frame_bgr.copy(),
                "landmarks": None,
                "keypoints": None,
                "scores": None,
                "orientation": "not_front",
                "orientation_debug": {
                    "reason": "no_best_person",
                    "orientation_raw": "not_front",
                    "orientation_smoothed": "not_front",
                    "front_score": 0,
                },
            }

        keypoints = np.array(best_person["keypoints"], dtype=float).squeeze()
        scores = np.array(best_person["keypoint_scores"], dtype=float).squeeze()

        raw_orientation, raw_measurement_allowed, orientation_debug = estimate_orientation(
            keypoints,
            scores,
            body_threshold=self.thr,
            face_threshold=self.face_thr
        )

        smoothed_orientation = self._smooth_orientation(raw_orientation)
        measurement_allowed = smoothed_orientation == "front"

        orientation_debug["orientation_smoothed"] = smoothed_orientation
        orientation_debug["front_counter"] = int(self.front_counter)
        orientation_debug["not_front_counter"] = int(self.not_front_counter)

        landmarks = compute_landmarks(
            keypoints,
            scores,
            threshold=self.thr,
            thyroid_alpha=self.thyroid_alpha,
            prostate_offset_alpha=self.prostate_offset_alpha,
            measurement_allowed=measurement_allowed,
        )

        if visualizations is not None and len(visualizations) > 0:
            vis_img = ensure_bgr(visualizations[0])
        else:
            vis_img = frame_bgr.copy()

        final_img = draw_interest_points(
            vis_img,
            landmarks,
            orientation=smoothed_orientation,
            orientation_debug=orientation_debug
        )

        return {
            "success": True,
            "image": final_img,
            "landmarks": landmarks,
            "keypoints": keypoints,
            "scores": scores,
            "orientation": smoothed_orientation,
            "orientation_debug": orientation_debug,
        }


# ============================================================
# EJECUCIÓN SOBRE UNA IMAGEN
# ============================================================

import argparse
import json
import os


def point_or_none(p):
    if p is None:
        return None
    return {"x": float(p[0]), "y": float(p[1])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTMPose landmarks inference on one image")
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Path to output annotated image")
    parser.add_argument("--json_output", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g. cuda:0 or cpu")
    parser.add_argument("--thr", type=float, default=0.3, help="Body keypoint confidence threshold")
    parser.add_argument("--face_thr", type=float, default=0.5, help="Face keypoint confidence threshold")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.json_output is not None:
        os.makedirs(os.path.dirname(args.json_output), exist_ok=True)

    img = cv2.imread(args.input_image)

    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {args.input_image}")

    engine = LandmarksEngine(
        device=args.device,
        thr=args.thr,
        face_thr=args.face_thr,
        front_frames_required=1,
        not_front_frames_required=1,
    )

    result = engine.process_frame(img)

    cv2.imwrite(args.output, result["image"])

    if args.json_output is not None:
        payload = {
            "success": bool(result["success"]),
            "orientation": result["orientation"],
            "orientation_debug": result["orientation_debug"],
            "landmarks": None,
            "keypoints": None,
            "scores": None,
        }

        if result["landmarks"] is not None:
            payload["landmarks"] = {
                "neck_base": point_or_none(result["landmarks"]["neck_base"]),
                "thyroid": point_or_none(result["landmarks"]["thyroid"]),
                "pelvis": point_or_none(result["landmarks"]["pelvis"]),
                "prostate": point_or_none(result["landmarks"]["prostate"]),
                "measurement_allowed": bool(result["landmarks"]["measurement_allowed"]),
            }

        if result["keypoints"] is not None:
            payload["keypoints"] = [
                {"x": float(p[0]), "y": float(p[1])} for p in result["keypoints"]
            ]

        if result["scores"] is not None:
            payload["scores"] = [float(s) for s in result["scores"]]

        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Imagen guardada en: {args.output}")

    if args.json_output is not None:
        print(f"JSON guardado en: {args.json_output}")
