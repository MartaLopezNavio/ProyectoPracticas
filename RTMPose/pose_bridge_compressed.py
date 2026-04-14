import os
import json
import socket
import struct
import pickle
import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray, String


HOST = "127.0.0.1"
PORT = 5055

OUTPUT_DIR = "mobile_output"
LATEST_FRAME_PATH = os.path.join(OUTPUT_DIR, "latest_frame.jpg")
LATEST_LANDMARKS_PATH = os.path.join(OUTPUT_DIR, "latest_landmarks.json")


def recv_all(sock, size):
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data


def recv_message(sock):
    header = recv_all(sock, 4)
    if header is None:
        return None
    msg_len = struct.unpack("!I", header)[0]
    payload = recv_all(sock, msg_len)
    return payload


def send_message(sock, payload_bytes):
    header = struct.pack("!I", len(payload_bytes))
    sock.sendall(header + payload_bytes)


def decode_jpg_bytes_to_bgr(jpg_bytes):
    arr = np.frombuffer(jpg_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class PoseBridgeCompressedNode(Node):
    def __init__(self):
        super().__init__("pose_bridge_compressed_node")

        self.declare_parameter("input_topic", "/camera/image/compressed")
        self.declare_parameter("output_image_topic", "/pose_app/image/compressed")
        self.declare_parameter("publish_ros_topics", True)
        self.declare_parameter("show_window", True)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_image_topic = self.get_parameter("output_image_topic").value
        self.publish_ros_topics = bool(self.get_parameter("publish_ros_topics").value)
        self.show_window = bool(self.get_parameter("show_window").value)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if self.publish_ros_topics:
            self.image_pub = self.create_publisher(
                CompressedImage, self.output_image_topic, 10
            )
            self.thyroid_pub = self.create_publisher(
                PointStamped, "/pose_app/thyroid", 10
            )
            self.prostate_pub = self.create_publisher(
                PointStamped, "/pose_app/prostate", 10
            )
            self.keypoints_pub = self.create_publisher(
                Float64MultiArray, "/pose_app/keypoints", 10
            )
            self.all_landmarks_pub = self.create_publisher(
                Float64MultiArray, "/pose_app/all_landmarks", 10
            )
            self.debug_pub = self.create_publisher(
                String, "/pose_app/debug", 10
            )

        self.sub = self.create_subscription(
            CompressedImage,
            self.input_topic,
            self.image_callback,
            10
        )

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((HOST, PORT))

        self.get_logger().info(f"Conectado a landmarks_server en {HOST}:{PORT}")
        self.get_logger().info(f"Escuchando imágenes en {self.input_topic}")
        self.get_logger().info(f"Guardando resultados en {OUTPUT_DIR}/")

    def publish_point(self, pub, point_xy, frame_id):
        if point_xy is None:
            return
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.point.x = float(point_xy[0])
        msg.point.y = float(point_xy[1])
        msg.point.z = 0.0
        pub.publish(msg)

    def bgr_to_compressed_msg(self, img_bgr, frame_id="camera"):
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.format = "jpeg"
        success, encoded = cv2.imencode(".jpg", img_bgr)
        if not success:
            return None
        msg.data = encoded.tobytes()
        return msg

    def save_json(self, landmarks, keypoints, scores, frame_id):
        def point_or_none(p):
            if p is None:
                return None
            return {"x": float(p[0]), "y": float(p[1])}

        payload = {
            "valid": landmarks is not None,
            "frame_id": frame_id,
            "thyroid": None,
            "prostate": None,
            "keypoints": [],
        }

        if landmarks is not None:
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

        with open(LATEST_LANDMARKS_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def save_image(self, img_bgr):
        cv2.imwrite(LATEST_FRAME_PATH, img_bgr)

    def image_callback(self, msg):
        try:
            req = {"image_jpg": bytes(msg.data)}
            payload = pickle.dumps(req)

            send_message(self.sock, payload)
            response_bytes = recv_message(self.sock)

            if response_bytes is None:
                self.get_logger().error("No se recibió respuesta del servidor.")
                return

            response = pickle.loads(response_bytes)

            if not response.get("success", False):
                self.get_logger().warning(f"Respuesta no válida: {response}")
                return

            image_jpg = response.get("image_jpg", None)
            landmarks = response.get("landmarks", None)
            keypoints = response.get("keypoints", None)
            scores = response.get("scores", None)

            final_img = None
            if image_jpg is not None:
                final_img = decode_jpg_bytes_to_bgr(image_jpg)

            if final_img is not None:
                self.save_image(final_img)

                if self.publish_ros_topics:
                    out_msg = self.bgr_to_compressed_msg(
                        final_img,
                        frame_id=msg.header.frame_id
                    )
                    if out_msg is not None:
                        out_msg.header.stamp = msg.header.stamp
                        self.image_pub.publish(out_msg)

                if self.show_window:
                    cv2.imshow("Pose bridge compressed", final_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        rclpy.shutdown()

            self.save_json(landmarks, keypoints, scores, msg.header.frame_id)

            if self.publish_ros_topics and landmarks is not None:
                self.publish_point(
                    self.thyroid_pub,
                    landmarks.get("thyroid"),
                    msg.header.frame_id
                )
                self.publish_point(
                    self.prostate_pub,
                    landmarks.get("prostate"),
                    msg.header.frame_id
                )

                def xy_or_nan(p):
                    if p is None:
                        return [float("nan"), float("nan")]
                    return [float(p[0]), float(p[1])]

                lm_msg = Float64MultiArray()
                lm_msg.data = (
                    xy_or_nan(landmarks.get("thyroid")) +
                    xy_or_nan(landmarks.get("prostate"))
                )
                self.all_landmarks_pub.publish(lm_msg)

                dbg = String()
                dbg.data = str(landmarks)
                self.debug_pub.publish(dbg)

            if self.publish_ros_topics and keypoints is not None and scores is not None:
                kp_msg = Float64MultiArray()
                data = []
                for kp, sc in zip(keypoints, scores):
                    data.extend([float(kp[0]), float(kp[1]), float(sc)])
                kp_msg.data = data
                self.keypoints_pub.publish(kp_msg)

        except Exception as e:
            self.get_logger().error(f"Error procesando imagen comprimida: {e}")

    def destroy_node(self):
        try:
            if hasattr(self, "sock") and self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PoseBridgeCompressedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
