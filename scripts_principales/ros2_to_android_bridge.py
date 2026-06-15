#!/usr/bin/env python3

import json
import socket
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Ros2ToAndroidBridge(Node):
    def __init__(self):
        super().__init__("ros2_to_android_bridge")

        # Parámetros configurables
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 9999)
        self.declare_parameter("topic", "/pose_app/distance/action")
        self.declare_parameter("send_period", 0.2)

        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.topic = self.get_parameter("topic").value
        self.send_period = float(self.get_parameter("send_period").value)

        self.latest_action = "SIN_ACCION"
        self.latest_stamp = time.time()

        self.tcp_clients = []
        self.tcp_clients_lock = threading.Lock()

        self.sub = self.create_subscription(
            String,
            self.topic,
            self.action_callback,
            10
        )

        self.timer = self.create_timer(
            self.send_period,
            self.send_latest_action
        )

        self.server_thread = threading.Thread(
            target=self.tcp_server,
            daemon=True
        )
        self.server_thread.start()

        self.get_logger().info("======================================")
        self.get_logger().info("ROS2 → Android Bridge iniciado")
        self.get_logger().info(f"Escuchando topic ROS2: {self.topic}")
        self.get_logger().info(f"Servidor TCP: {self.host}:{self.port}")
        self.get_logger().info("La app Android debe conectarse a la IP del PC y puerto 9999")
        self.get_logger().info("======================================")

    def action_callback(self, msg):
        action = msg.data.strip()

        if action == "":
            action = "SIN_ACCION"

        self.latest_action = action
        self.latest_stamp = time.time()

        self.get_logger().info(f"Acción recibida desde ROS2: {action}")

    def tcp_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            server.bind((self.host, self.port))
            server.listen(5)
        except Exception as e:
            self.get_logger().error(f"No se pudo abrir servidor TCP: {e}")
            return

        while rclpy.ok():
            try:
                client_socket, addr = server.accept()
                client_socket.settimeout(1.0)

                with self.tcp_clients_lock:
                    self.tcp_clients.append(client_socket)

                self.get_logger().info(f"Android conectado desde {addr}")

                # Enviar inmediatamente el último estado al conectar
                self.send_to_client(client_socket)

            except Exception as e:
                self.get_logger().warn(f"Error aceptando cliente TCP: {e}")

    def send_latest_action(self):
        disconnected = []

        with self.tcp_clients_lock:
            for client in self.tcp_clients:
                ok = self.send_to_client(client)
                if not ok:
                    disconnected.append(client)

            for client in disconnected:
                try:
                    client.close()
                except Exception:
                    pass

                if client in self.tcp_clients:
                    self.tcp_clients.remove(client)

        if disconnected:
            self.get_logger().warn("Cliente Android desconectado")

    def send_to_client(self, client):
        payload = {
            "action": self.latest_action,
            "timestamp": self.latest_stamp
        }

        try:
            data = json.dumps(payload) + "\n"
            client.sendall(data.encode("utf-8"))
            return True

        except Exception:
            return False


def main(args=None):
    rclpy.init(args=args)

    node = Ros2ToAndroidBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

