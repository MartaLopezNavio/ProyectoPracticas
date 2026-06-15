import os
import json
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray, String, Bool

OUTPUT_DIR = "mobile_output"
LATEST_FRAME_PATH = os.path.join(OUTPUT_DIR, "latest_frame.jpg")
LATEST_LANDMARKS_PATH = os.path.join(OUTPUT_DIR, "latest_landmarks.json")


class MobilePosePublisher(Node):
    def __init__(self):
        super().__init__("mobile_pose_publisher")

        self.image_pub = self.create_publisher(CompressedImage, "/pose_app/image/compressed", 10)
        self.thyroid_pub = self.create_publisher(PointStamped, "/pose_app/thyroid", 10)
        self.prostate_pub = self.create_publisher(PointStamped, "/pose_app/prostate", 10)
        self.keypoints_pub = self.create_publisher(Float64MultiArray, "/pose_app/keypoints", 10)
        self.all_landmarks_pub = self.create_publisher(Float64MultiArray, "/pose_app/all_landmarks", 10)
        self.debug_pub = self.create_publisher(String, "/pose_app/debug", 10)
        self.orientation_pub = self.create_publisher(String, "/pose_app/orientation", 10)
        self.measurement_allowed_pub = self.create_publisher(Bool, "/pose_app/measurement_allowed", 10)

        self.last_frame_mtime = None
        self.last_json_mtime = None

        self.timer = self.create_timer(0.2, self.timer_callback)

    def publish_point(self, pub, point_xy, frame_id="mobile_camera"):
        if point_xy is None:
            return
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.point.x = float(point_xy["x"])
        msg.point.y = float(point_xy["y"])
        msg.point.z = 0.0
        pub.publish(msg)

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

    def publish_landmarks(self):
        with open(LATEST_LANDMARKS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        thyroid = data.get("thyroid")
        prostate = data.get("prostate")
        orientation = data.get("orientation", "unknown")
        measurement_allowed = bool(data.get("measurement_allowed", False))

        # Publicar landmarks principales
        self.publish_point(self.thyroid_pub, thyroid)
        self.publish_point(self.prostate_pub, prostate)

        # Publicar landmarks agrupados
        lm_msg = Float64MultiArray()
        lm_msg.data = [
            thyroid["x"] if thyroid else float("nan"),
            thyroid["y"] if thyroid else float("nan"),
            prostate["x"] if prostate else float("nan"),
            prostate["y"] if prostate else float("nan"),
        ]
        self.all_landmarks_pub.publish(lm_msg)

        # Publicar keypoints completos
        keypoints = data.get("keypoints", [])
        kp_msg = Float64MultiArray()
        kp_data = []
        for kp in keypoints:
            kp_data.extend([kp["x"], kp["y"], kp["score"]])
        kp_msg.data = kp_data
        self.keypoints_pub.publish(kp_msg)

        # Publicar orientación
        orientation_msg = String()
        orientation_msg.data = orientation
        self.orientation_pub.publish(orientation_msg)

        # Publicar permiso de medición
        measurement_msg = Bool()
        measurement_msg.data = measurement_allowed
        self.measurement_allowed_pub.publish(measurement_msg)

        # Publicar debug completo
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
