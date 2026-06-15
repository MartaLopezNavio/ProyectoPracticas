import os
import json
import cv2
from pathlib import Path

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray, String, Bool, Float32


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "mobile_output"
LATEST_FRAME_PATH = OUTPUT_DIR / "latest_frame.jpg"
LATEST_LANDMARKS_PATH = OUTPUT_DIR / "latest_landmarks.json"

TARGET_NAMES = ["thyroid", "prostate"]


class MobilePosePublisher(Node):
    def __init__(self):
        super().__init__("mobile_pose_publisher")

        # =====================================================
        # IMAGEN
        # =====================================================

        self.image_pub = self.create_publisher(
            CompressedImage,
            "/pose_app/image/compressed",
            10
        )

        # =====================================================
        # LANDMARKS ANATÓMICOS PRINCIPALES
        # =====================================================

        self.thyroid_pub = self.create_publisher(
            PointStamped,
            "/pose_app/thyroid",
            10
        )

        self.prostate_pub = self.create_publisher(
            PointStamped,
            "/pose_app/prostate",
            10
        )

        # =====================================================
        # DATOS AGRUPADOS
        # =====================================================

        self.keypoints_pub = self.create_publisher(
            Float64MultiArray,
            "/pose_app/keypoints",
            10
        )

        self.all_landmarks_pub = self.create_publisher(
            Float64MultiArray,
            "/pose_app/all_landmarks",
            10
        )

        # =====================================================
        # ESTADO GENERAL
        # =====================================================

        self.debug_pub = self.create_publisher(
            String,
            "/pose_app/debug",
            10
        )

        self.orientation_pub = self.create_publisher(
            String,
            "/pose_app/orientation",
            10
        )

        self.measurement_allowed_pub = self.create_publisher(
            Bool,
            "/pose_app/measurement_allowed",
            10
        )

        # =====================================================
        # DISTANCIA PRINCIPAL
        # =====================================================

        self.distance_raw_depth_pub = self.create_publisher(
            Float32,
            "/pose_app/distance/raw_depth",
            10
        )

        self.distance_smooth_depth_pub = self.create_publisher(
            Float32,
            "/pose_app/distance/smooth_depth",
            10
        )

        self.distance_state_pub = self.create_publisher(
            String,
            "/pose_app/distance/state",
            10
        )

        self.distance_instant_state_pub = self.create_publisher(
            String,
            "/pose_app/distance/instant_state",
            10
        )

        self.distance_action_pub = self.create_publisher(
            String,
            "/pose_app/distance/action",
            10
        )

        self.distance_target_pub = self.create_publisher(
            PointStamped,
            "/pose_app/distance/target",
            10
        )

        self.distance_target_name_pub = self.create_publisher(
            String,
            "/pose_app/distance/target_name",
            10
        )

        self.distance_measurement_allowed_pub = self.create_publisher(
            Bool,
            "/pose_app/distance/measurement_allowed",
            10
        )

        self.distance_near_threshold_pub = self.create_publisher(
            Float32,
            "/pose_app/distance/near_threshold",
            10
        )

        # =====================================================
        # DISTANCIA SEPARADA PARA TIROIDES Y PRÓSTATA
        # =====================================================

        self.target_distance_pubs = {}

        for target_name in TARGET_NAMES:
            base = f"/pose_app/distance/{target_name}"

            self.target_distance_pubs[target_name] = {
                "available": self.create_publisher(
                    Bool,
                    f"{base}/available",
                    10
                ),
                "point": self.create_publisher(
                    PointStamped,
                    f"{base}/point",
                    10
                ),
                "raw_depth": self.create_publisher(
                    Float32,
                    f"{base}/raw_depth",
                    10
                ),
                "smooth_depth": self.create_publisher(
                    Float32,
                    f"{base}/smooth_depth",
                    10
                ),
                "instant_state": self.create_publisher(
                    String,
                    f"{base}/instant_state",
                    10
                ),
                "state": self.create_publisher(
                    String,
                    f"{base}/state",
                    10
                ),
                "action": self.create_publisher(
                    String,
                    f"{base}/action",
                    10
                ),
            }

        self.last_frame_mtime = None
        self.last_json_mtime = None

        self.timer = self.create_timer(0.2, self.timer_callback)

        self.get_logger().info("MobilePosePublisher iniciado.")
        self.get_logger().info("Publicando imagen, landmarks, orientación, distancia y acción en /pose_app/...")

    # =====================================================
    # FUNCIONES AUXILIARES DE PUBLICACIÓN
    # =====================================================

    def publish_point(self, pub, point_xy, frame_id="mobile_camera"):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        if point_xy is None:
            msg.point.x = float("nan")
            msg.point.y = float("nan")
            msg.point.z = float("nan")
        else:
            try:
                msg.point.x = float(point_xy["x"])
                msg.point.y = float(point_xy["y"])
                msg.point.z = 0.0
            except Exception:
                msg.point.x = float("nan")
                msg.point.y = float("nan")
                msg.point.z = float("nan")

        pub.publish(msg)

    def publish_float32(self, pub, value):
        msg = Float32()

        if value is None:
            msg.data = float("nan")
        else:
            try:
                msg.data = float(value)
            except Exception:
                msg.data = float("nan")

        pub.publish(msg)

    def publish_string(self, pub, value):
        msg = String()
        msg.data = str(value) if value is not None else ""
        pub.publish(msg)

    def publish_bool(self, pub, value):
        msg = Bool()
        msg.data = bool(value)
        pub.publish(msg)

    # =====================================================
    # TIMER
    # =====================================================

    def timer_callback(self):
        if os.path.exists(LATEST_FRAME_PATH):
            mtime = os.path.getmtime(LATEST_FRAME_PATH)

            if self.last_frame_mtime != mtime:
                self.last_frame_mtime = mtime
                self.publish_image()

        if os.path.exists(LATEST_LANDMARKS_PATH):
            mtime = os.path.getmtime(LATEST_LANDMARKS_PATH)

            if self.last_json_mtime != mtime:
                self.last_json_mtime = mtime
                self.publish_landmarks()

    # =====================================================
    # PUBLICAR IMAGEN
    # =====================================================

    def publish_image(self):
        img = cv2.imread(LATEST_FRAME_PATH)

        if img is None:
            return

        ok, enc = cv2.imencode(".jpg", img)

        if not ok:
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "mobile_camera"
        msg.format = "jpeg"
        msg.data = enc.tobytes()

        self.image_pub.publish(msg)

    # =====================================================
    # PUBLICAR DISTANCIA DE CADA TARGET
    # =====================================================

    def publish_target_distance(self, target_name, target_data):
        if target_name not in self.target_distance_pubs:
            return

        pubs = self.target_distance_pubs[target_name]

        if target_data is None or not isinstance(target_data, dict):
            target_data = {}

        available = bool(target_data.get("available", False))
        point = target_data.get("point", None)
        raw_depth = target_data.get("raw_depth", None)
        smooth_depth = target_data.get("smooth_depth", None)
        instant_state = target_data.get("instant_state", "SIN_MEDIDA")
        stable_state = target_data.get("stable_state", "SIN_MEDIDA")
        action = target_data.get("action", "SIN_ACCION")

        self.publish_bool(
            pubs["available"],
            available
        )

        self.publish_point(
            pubs["point"],
            point
        )

        self.publish_float32(
            pubs["raw_depth"],
            raw_depth
        )

        self.publish_float32(
            pubs["smooth_depth"],
            smooth_depth
        )

        self.publish_string(
            pubs["instant_state"],
            instant_state
        )

        self.publish_string(
            pubs["state"],
            stable_state
        )

        self.publish_string(
            pubs["action"],
            action
        )

    # =====================================================
    # PUBLICAR DISTANCIA PRINCIPAL
    # =====================================================

    def publish_distance(self, data):
        distance = data.get("distance", {})

        if distance is None or not isinstance(distance, dict):
            distance = {}

        raw_depth = distance.get("raw_depth", None)
        smooth_depth = distance.get("smooth_depth", None)
        instant_state = distance.get("instant_state", "SIN_MEDIDA")
        stable_state = distance.get("stable_state", "SIN_MEDIDA")
        action = distance.get("action", "SIN_ACCION")
        target_point = distance.get("target_point", None)
        target_name = distance.get("target", "")
        near_threshold = distance.get("near_threshold", None)

        distance_measurement_allowed = bool(
            distance.get("measurement_allowed", False)
        )

        self.publish_float32(
            self.distance_raw_depth_pub,
            raw_depth
        )

        self.publish_float32(
            self.distance_smooth_depth_pub,
            smooth_depth
        )

        self.publish_string(
            self.distance_instant_state_pub,
            instant_state
        )

        self.publish_string(
            self.distance_state_pub,
            stable_state
        )

        self.publish_string(
            self.distance_action_pub,
            action
        )

        self.publish_point(
            self.distance_target_pub,
            target_point
        )

        self.publish_string(
            self.distance_target_name_pub,
            target_name
        )

        self.publish_bool(
            self.distance_measurement_allowed_pub,
            distance_measurement_allowed
        )

        self.publish_float32(
            self.distance_near_threshold_pub,
            near_threshold
        )

        targets = distance.get("targets", {})

        if not isinstance(targets, dict):
            targets = {}

        for name in TARGET_NAMES:
            self.publish_target_distance(
                name,
                targets.get(name, {})
            )

    # =====================================================
    # PUBLICAR LANDMARKS
    # =====================================================

    def publish_landmarks(self):
        try:
            with open(LATEST_LANDMARKS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

        except Exception as e:
            self.get_logger().warn(f"No se pudo leer el JSON de landmarks: {e}")
            return

        thyroid = data.get("thyroid")
        prostate = data.get("prostate")
        orientation = data.get("orientation", "unknown")
        measurement_allowed = bool(data.get("measurement_allowed", False))

        # Landmarks principales
        self.publish_point(self.thyroid_pub, thyroid)
        self.publish_point(self.prostate_pub, prostate)

        # Landmarks agrupados
        lm_msg = Float64MultiArray()
        lm_msg.data = [
            thyroid["x"] if thyroid else float("nan"),
            thyroid["y"] if thyroid else float("nan"),
            prostate["x"] if prostate else float("nan"),
            prostate["y"] if prostate else float("nan"),
        ]
        self.all_landmarks_pub.publish(lm_msg)

        # Keypoints completos
        keypoints = data.get("keypoints", [])

        kp_msg = Float64MultiArray()
        kp_data = []

        for kp in keypoints:
            kp_data.extend([
                float(kp["x"]),
                float(kp["y"]),
                float(kp["score"]),
            ])

        kp_msg.data = kp_data
        self.keypoints_pub.publish(kp_msg)

        # Orientación
        self.publish_string(
            self.orientation_pub,
            orientation
        )

        # Permiso de medición general
        self.publish_bool(
            self.measurement_allowed_pub,
            measurement_allowed
        )

        # Distancia
        self.publish_distance(data)

        # Debug completo
        dbg = String()
        dbg.data = json.dumps(data)
        self.debug_pub.publish(dbg)


def main(args=None):
    rclpy.init(args=args)

    node = MobilePosePublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
