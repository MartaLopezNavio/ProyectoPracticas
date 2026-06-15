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

    # 1) nose must be between both eyes in X
    nose_between_eyes = min(left_eye[0], right_eye[0]) <= nose[0] <= max(left_eye[0], right_eye[0])

    # 2) eyes roughly at the same height
    eye_y_diff = abs(left_eye[1] - right_eye[1])
    eyes_level = eye_y_diff < 0.08 * shoulder_width

    # 3) nose roughly centered relative to shoulder midpoint
    nose_center_offset = abs(nose[0] - shoulder_mid[0])
    nose_centered = nose_center_offset < 0.18 * shoulder_width

    # 4) eyes above shoulders
    shoulder_y = (left_sh[1] + right_sh[1]) / 2.0
    eyes_above_shoulders = left_eye[1] < shoulder_y and right_eye[1] < shoulder_y

    # 5) eye distance must be reasonable relative to shoulder width
    face_ratio = eye_dist / shoulder_width
    face_ratio_ok = 0.08 < face_ratio < 0.55

    # 6) nose should be similarly spaced from both eyes
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

    color_orientation = (0, 255, 0) if orientation == "front" else (0, 0, 255)

    cv2.putText(
        out,
        f"orientation: {orientation}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color_orientation,
        2,
    )

    if orientation_debug is not None:
        score = orientation_debug.get("front_score", 0)
        cv2.putText(
            out,
            f"front_score: {score}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
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

        # temporal smoothing
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
