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


def estimate_orientation(keypoints, scores, threshold=0.3):
    nose_conf = scores[COCO_KEYPOINTS["nose"]]
    left_eye_conf = scores[COCO_KEYPOINTS["left_eye"]]
    right_eye_conf = scores[COCO_KEYPOINTS["right_eye"]]
    left_sh_conf = scores[COCO_KEYPOINTS["left_shoulder"]]
    right_sh_conf = scores[COCO_KEYPOINTS["right_shoulder"]]
    left_hip_conf = scores[COCO_KEYPOINTS["left_hip"]]
    right_hip_conf = scores[COCO_KEYPOINTS["right_hip"]]

    face_count = sum([
        is_valid(nose_conf, threshold),
        is_valid(left_eye_conf, threshold),
        is_valid(right_eye_conf, threshold),
    ])

    upper_body_count = sum([
        is_valid(left_sh_conf, threshold),
        is_valid(right_sh_conf, threshold),
        is_valid(left_hip_conf, threshold),
        is_valid(right_hip_conf, threshold),
    ])

    if face_count >= 3:
        return "front", True

    if face_count == 0 and upper_body_count >= 2:
        return "back", False

    if face_count >= 1 and upper_body_count >= 2:
        return "side_or_partial", False

    return "unknown", False


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

    # La tiroides se calcula siempre que haya hombros y nariz válidos
    if neck_base is not None and is_valid(nose_conf, threshold):
        thyroid = point_between(neck_base, nose, thyroid_alpha)

    # La próstata se calcula siempre que haya pelvis válida
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


def draw_interest_points(img, landmarks, orientation="unknown"):
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

    color_orientation = (0, 255, 0)
    if orientation == "back":
        color_orientation = (0, 0, 255)
    elif orientation == "side_or_partial":
        color_orientation = (0, 255, 255)
    elif orientation == "unknown":
        color_orientation = (200, 200, 200)

    cv2.putText(
        out,
        f"orientation: {orientation}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color_orientation,
        2,
    )

    return out

class LandmarksEngine:
    def __init__(self, device="cuda:0", thr=0.3, thyroid_alpha=0.25, prostate_offset_alpha=0.15):
        self.device = device
        self.thr = thr
        self.thyroid_alpha = thyroid_alpha
        self.prostate_offset_alpha = prostate_offset_alpha
        self.inferencer = MMPoseInferencer(pose2d="human", device=device)

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
                "orientation": "unknown",
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
                "orientation": "unknown",
            }

        keypoints = np.array(best_person["keypoints"], dtype=float).squeeze()
        scores = np.array(best_person["keypoint_scores"], dtype=float).squeeze()

        orientation, measurement_allowed = estimate_orientation(
            keypoints,
            scores,
            threshold=self.thr
        )

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

        final_img = draw_interest_points(vis_img, landmarks, orientation=orientation)

        return {
            "success": True,
            "image": final_img,
            "landmarks": landmarks,
            "keypoints": keypoints,
            "scores": scores,
            "orientation": orientation,
        }
