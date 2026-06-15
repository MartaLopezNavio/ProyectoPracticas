#!/usr/bin/env python3

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Int8MultiArray
from geometry_msgs.msg import Twist


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Robot Monitor</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
    h1 { margin-bottom: 10px; }
    .card {
      background: white; padding: 16px; margin-bottom: 12px;
      border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .label { font-weight: bold; }
    pre {
      background: #111; color: #0f0; padding: 12px; border-radius: 8px;
      overflow-x: auto;
    }
  </style>
</head>
<body>
  <h1>Robot Monitor</h1>

  <div class="card"><span class="label">Robot state:</span> <span id="robot_state">-</span></div>
  <div class="card"><span class="label">Online motors:</span> <span id="online_id">-</span></div>
  <div class="card"><span class="label">Last cmd_vel:</span> <pre id="cmd_vel">-</pre></div>

  <script>
    async function refreshState() {
      try {
        const response = await fetch('/state');
        const data = await response.json();

        document.getElementById('robot_state').textContent = data.robot_state;
        document.getElementById('online_id').textContent = JSON.stringify(data.online_id);
        document.getElementById('cmd_vel').textContent = JSON.stringify(data.cmd_vel, null, 2);
      } catch (e) {
        console.error(e);
      }
    }

    setInterval(refreshState, 300);
    refreshState();
  </script>
</body>
</html>
"""


class SharedState:
    def __init__(self):
        self.robot_state = "INIT"
        self.online_id = []
        self.cmd_vel = {
            "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
        }


class RobotWebServer(Node):
    def __init__(self):
        super().__init__("robot_web_server")

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8080)

        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)

        self.state = SharedState()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(String, "/robot_state", self.robot_state_cb, qos)
        self.create_subscription(Int8MultiArray, "/ddsm115/online_id", self.online_id_cb, qos)
        self.create_subscription(Twist, "/cmd_vel", self.cmd_vel_cb, qos)

        self.http_thread = threading.Thread(target=self.start_http_server, daemon=True)
        self.http_thread.start()

        self.get_logger().info(f"robot_web_server running on http://{self.host}:{self.port}")

    def robot_state_cb(self, msg: String):
        self.state.robot_state = msg.data

    def online_id_cb(self, msg: Int8MultiArray):
        self.state.online_id = list(msg.data)

    def cmd_vel_cb(self, msg: Twist):
        self.state.cmd_vel = {
            "linear": {
                "x": msg.linear.x,
                "y": msg.linear.y,
                "z": msg.linear.z,
            },
            "angular": {
                "x": msg.angular.x,
                "y": msg.angular.y,
                "z": msg.angular.z,
            },
        }

    def start_http_server(self):
        shared_state = self.state

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(HTML_PAGE.encode("utf-8"))
                    return

                if self.path == "/state":
                    payload = {
                        "robot_state": shared_state.robot_state,
                        "online_id": shared_state.online_id,
                        "cmd_vel": shared_state.cmd_vel,
                    }
                    data = json.dumps(payload).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return

                self.send_response(404)
                self.end_headers()

            def log_message(self, format, *args):
                return

        httpd = HTTPServer((self.host, self.port), Handler)
        httpd.serve_forever()


def main(args=None):
    rclpy.init(args=args)
    node = RobotWebServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
