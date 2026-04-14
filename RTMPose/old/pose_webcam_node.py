#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray, String

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


def compute_landmarks(keypoints, scores, threshold=0.3, thyroid_alpha=0.25, prostate_offset_alpha=0.15):
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

    # Tiroides aproximada: un punto entre base del cuello y nariz
    if neck_base is not None and is_valid(nose_conf, threshold):
        thyroid = point_between(neck_base, nose, thyroid_alpha)

    # Próstata aproximada: desde pelvis un poco hacia abajo
    # (solo referencia externa)
    if pelvis is not None:
        prostate = [pelvis[0], pelvis[1] + prostate_offset_alpha * abs(r_hip[0] - l_hip[0])]

    return {
        "neck_base": neck_base,
        "thyroid": thyroid,
        "pelvis": pelvis,
        "prostate": prostate,
    }


def draw_only_interest_points(img, landmarks):
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
            "prostate_approx",
            (int(prostate[0]) + 8, int(prostate[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
        )

    # Línea auxiliar pelvis -> próstata aprox, solo si quieres dejar clara la estimación
    if pelvis is not None and prostate is not None:
        cv2.line(
            out,
            (int(pelvis[0]), int(pelvis[1])),
            (int(prostate[0]), int(prostate[1])),
            (255, 0, 255),
            2,
        )

    return out


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


class PoseWebcamNode(Node):
    def __init__(self):
        super().__init__("pose_webcam_node")

        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("thr", 0.3)
        self.declare_parameter("thyroid_alpha", 0.25)
        self.declare_parameter("prostate_offset_alpha", 0.15)
        self.declare_parameter("timer_hz", 10.0)
        self.declare_parameter("show_window", True)

        self.device = self.get_parameter("device").value
        self.camera_index = int(self.get_parameter("camera_index").value)
        self.thr = float(self.get_parameter("thr").value)
        self.thyroid_alpha = float(self.get_parameter("thyroid_alpha").value)
        self.prostate_offset_alpha = float(self.get_parameter("prostate_offset_alpha").value)
        self.timer_hz = float(self.get_parameter("timer_hz").value)
        self.show_window = bool(self.get_parameter("show_window").value)

        self.image_pub = self.create_publisher(CompressedImage, "/pose_app/image/compressed", 10)
        self.thyroid_pub = self.create_publisher(PointStamped, "/pose_app/thyroid", 10)
        self.prostate_pub = self.create_publisher(PointStamped, "/pose_app/prostate", 10)
        self.keypoints_pub = self.create_publisher(Float64MultiArray, "/pose_app/keypoints", 10)
        self.all_landmarks_pub = self.create_publisher(Float64MultiArray, "/pose_app/all_landmarks", 10)
        self.debug_pub = self.create_publisher(String, "/pose_app/debug", 10)

        self.get_logger().info(f"Cargando RTMPose en {self.device}...")
        self.inferencer = MMPoseInferencer(pose2d="human", device=self.device)

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la webcam con índice {self.camera_index}")

        self.timer = self.create_timer(1.0 / self.timer_hz, self.timer_callback)
        self.get_logger().info("Nodo listo. Publicando topics desde webcam.")

    def bgr_to_compressed(self, img):
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "webcam"
        msg.format = "jpeg"
        success, encoded = cv2.imencode(".jpg", img)
        if not success:
            return None
        msg.data = encoded.tobytes()
        return msg

    def publish_point(self, pub, point_xy):
        if point_xy is None:
            return
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "webcam"
        msg.point.x = float(point_xy[0])
        msg.point.y = float(point_xy[1])
        msg.point.z = 0.0
        pub.publish(msg)

    def publish_keypoints(self, keypoints, scores):
        msg = Float64MultiArray()
        data = []
        for kp, sc in zip(keypoints, scores):
            data.extend([float(kp[0]), float(kp[1]), float(sc)])
        msg.data = data
        self.keypoints_pub.publish(msg)

    def publish_all_landmarks(self, landmarks):
        def xy_or_nan(p):
            if p is None:
                return [float("nan"), float("nan")]
            return [float(p[0]), float(p[1])]

        msg = Float64MultiArray()
        data = []
        data.extend(xy_or_nan(landmarks["thyroid"]))
        data.extend(xy_or_nan(landmarks["prostate"]))
        msg.data = data
        self.all_landmarks_pub.publish(msg)

    def publish_debug(self, landmarks):
        msg = String()
        msg.data = str(landmarks)
        self.debug_pub.publish(msg)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("No se pudo leer frame de la webcam.")
            return

        try:
            result = next(self.inferencer(frame, return_vis=True))

            predictions = result.get("predictions", None)
            visualizations = result.get("visualization", None)

            if not predictions or not predictions[0]:
                final_img = frame.copy()
                out_msg = self.bgr_to_compressed(final_img)
                if out_msg is not None:
                    self.image_pub.publish(out_msg)
                if self.show_window:
                    cv2.imshow("RTMPose + thyroid + prostate", final_img)
                    cv2.waitKey(1)
                return

            people = predictions[0]
            best_person = pick_best_person(people)
            if best_person is None:
                return

            keypoints = np.array(best_person["keypoints"], dtype=float).squeeze()
            scores = np.array(best_person["keypoint_scores"], dtype=float).squeeze()

            landmarks = compute_landmarks(
                keypoints,
                scores,
                threshold=self.thr,
                thyroid_alpha=self.thyroid_alpha,
                prostate_offset_alpha=self.prostate_offset_alpha
            )

            if visualizations is not None and len(visualizations) > 0:
                vis_img = ensure_bgr(visualizations[0])
            else:
                vis_img = frame.copy()

            # Se deja el esqueleto de MMPose y se añaden solo tiroides/próstata
            final_img = draw_only_interest_points(vis_img, landmarks)

            out_msg = self.bgr_to_compressed(final_img)
            if out_msg is not None:
                self.image_pub.publish(out_msg)

            self.publish_point(self.thyroid_pub, landmarks["thyroid"])
            self.publish_point(self.prostate_pub, landmarks["prostate"])
            self.publish_keypoints(keypoints, scores)
            self.publish_all_landmarks(landmarks)
            self.publish_debug(landmarks)

            if self.show_window:
                cv2.imshow("RTMPose + thyroid + prostate", final_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.get_logger().info("Saliendo por teclado.")
                    rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error procesando frame: {e}")

    def destroy_node(self):
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PoseWebcamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
