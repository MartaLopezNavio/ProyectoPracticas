#!/usr/bin/env python3

import time
from typing import List

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String, Int16MultiArray, Int8MultiArray


class RobotMotorServer(Node):
    def __init__(self):
        super().__init__("robot_motor_server")
        self.get_logger().info("Starting robot_motor_server...")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("num_wheels", 3)
        self.declare_parameter("max_rpm", 200)
        self.declare_parameter("turn_gain", 100.0)
        self.declare_parameter("cmd_timeout_s", 1.0)
        self.declare_parameter("publish_debug_rpm", True)

        # Signo por rueda para corregir montaje físico.
        # -1 = invertida, 1 = normal
        # En tu caso la rueda 1 está al revés.
        self.declare_parameter("wheel_signs", [-1, 1, 1])

        self.num_wheels = int(self.get_parameter("num_wheels").value)
        self.max_rpm = int(self.get_parameter("max_rpm").value)
        self.turn_gain = float(self.get_parameter("turn_gain").value)
        self.cmd_timeout_s = float(self.get_parameter("cmd_timeout_s").value)
        self.publish_debug_rpm = bool(self.get_parameter("publish_debug_rpm").value)
        self.wheel_signs = list(self.get_parameter("wheel_signs").value)

        # Ajuste defensivo por si el array no coincide con num_wheels
        if len(self.wheel_signs) < self.num_wheels:
            self.get_logger().warn(
                f"wheel_signs tiene longitud {len(self.wheel_signs)} y num_wheels={self.num_wheels}. "
                "Se completará con 1."
            )
            self.wheel_signs += [1] * (self.num_wheels - len(self.wheel_signs))
        elif len(self.wheel_signs) > self.num_wheels:
            self.get_logger().warn(
                f"wheel_signs tiene más elementos que num_wheels. Se truncará a {self.num_wheels}."
            )
            self.wheel_signs = self.wheel_signs[:self.num_wheels]

        # -----------------------------
        # Internal state
        # -----------------------------
        self.enabled = True
        self.stop_requested = False
        self.last_cmd_time = 0.0

        self.last_cmd_vel = Twist()
        self.last_online_ids: List[int] = []
        self.last_rpm_fb: List[int] = []
        self.last_error_fb: List[int] = []

        self.current_rpm_cmd = [0] * self.num_wheels
        self.current_state = "INIT"

        # -----------------------------
        # Subscribers
        # -----------------------------
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_vel_callback,
            10,
        )

        self.stop_sub = self.create_subscription(
            Bool,
            "/stop_robot",
            self.stop_robot_callback,
            10,
        )

        self.rpm_fb_sub = self.create_subscription(
            Int16MultiArray,
            "/ddsm115/rpm_fb",
            self.rpm_fb_callback,
            10,
        )

        self.online_id_sub = self.create_subscription(
            Int8MultiArray,
            "/ddsm115/online_id",
            self.online_id_callback,
            10,
        )

        self.error_sub = self.create_subscription(
            Int8MultiArray,
            "/ddsm115/error",
            self.error_callback,
            10,
        )

        # -----------------------------
        # Publishers
        # -----------------------------
        self.rpm_cmd_pub = self.create_publisher(
            Int16MultiArray,
            "/ddsm115/rpm_cmd",
            10,
        )

        self.brake_pub = self.create_publisher(
            Bool,
            "/ddsm115/brake",
            10,
        )

        self.robot_state_pub = self.create_publisher(
            String,
            "/robot_state",
            10,
        )

        # Opcional, para depuración visual
        self.debug_rpm_pub = self.create_publisher(
            Int16MultiArray,
            "/robot_server/rpm_cmd_debug",
            10,
        )

        # -----------------------------
        # Timer loop
        # -----------------------------
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.current_state = "READY"
        self.publish_state()

        self.get_logger().info(f"num_wheels: {self.num_wheels}")
        self.get_logger().info(f"max_rpm: {self.max_rpm}")
        self.get_logger().info(f"turn_gain: {self.turn_gain}")
        self.get_logger().info(f"cmd_timeout_s: {self.cmd_timeout_s}")
        self.get_logger().info(f"wheel_signs: {self.wheel_signs}")

    # =========================================================
    # Callbacks
    # =========================================================
    def cmd_vel_callback(self, msg: Twist):
        self.last_cmd_vel = msg
        self.last_cmd_time = time.time()
        self.stop_requested = False

    def stop_robot_callback(self, msg: Bool):
        self.stop_requested = bool(msg.data)

    def rpm_fb_callback(self, msg: Int16MultiArray):
        self.last_rpm_fb = list(msg.data)

    def online_id_callback(self, msg: Int8MultiArray):
        self.last_online_ids = list(msg.data)

    def error_callback(self, msg: Int8MultiArray):
        self.last_error_fb = list(msg.data)

    # =========================================================
    # Main loop
    # =========================================================
    def timer_callback(self):
        now = time.time()

        # 1) Si se ha pedido stop, frena
        if self.stop_requested:
            self.current_rpm_cmd = [0] * self.num_wheels
            self.publish_rpm_cmd(self.current_rpm_cmd)
            self.publish_brake(True)
            self.current_state = "STOPPED"
            self.publish_state()
            return

        # 2) Si no llega cmd_vel en un tiempo, manda 0 y quita movimiento
        if (now - self.last_cmd_time) > self.cmd_timeout_s:
            self.current_rpm_cmd = [0] * self.num_wheels
            self.publish_brake(False)
            self.publish_rpm_cmd(self.current_rpm_cmd)
            self.current_state = "IDLE"
            self.publish_state()
            return

        # 3) Si hay errores de motor, pasa a ERROR
        if any(err != 0 for err in self.last_error_fb):
            self.current_rpm_cmd = [0] * self.num_wheels
            self.publish_rpm_cmd(self.current_rpm_cmd)
            self.publish_brake(True)
            self.current_state = "ERROR"
            self.publish_state()
            return

        # 4) Convertir cmd_vel a rpm de ruedas
        self.current_rpm_cmd = self.cmd_vel_to_rpm(self.last_cmd_vel)

        # 5) Publicar comando
        self.publish_brake(False)
        self.publish_rpm_cmd(self.current_rpm_cmd)

        # 6) Estado
        if any(abs(v) > 0 for v in self.current_rpm_cmd):
            self.current_state = "MOVING"
        else:
            self.current_state = "READY"

        self.publish_state()

    # =========================================================
    # Logic
    # =========================================================
    def cmd_vel_to_rpm(self, cmd: Twist) -> List[int]:
        """
        Conversión simple de /cmd_vel a rpm.

        Suposición actual:
        - rueda 1: izquierda
        - rueda 2: derecha
        - rueda 3: trasera/auxiliar

        Mezcla sencilla:
        left  = linear.x - angular.z * turn_gain
        right = linear.x + angular.z * turn_gain
        rear  = linear.x

        Después se aplica wheel_signs para corregir ruedas montadas al revés.
        """
        linear_term = cmd.linear.x * self.max_rpm
        turn_term = cmd.angular.z * self.turn_gain

        rpm = [0] * self.num_wheels

        if self.num_wheels >= 1:
            rpm[0] = int(linear_term - turn_term)

        if self.num_wheels >= 2:
            rpm[1] = int(linear_term + turn_term)

        if self.num_wheels >= 3:
            rpm[2] = int(linear_term)

        # Saturación
        rpm = [self.clamp(v, -self.max_rpm, self.max_rpm) for v in rpm]

        # Corrección por orientación física de las ruedas
        rpm = [rpm[i] * self.wheel_signs[i] for i in range(self.num_wheels)]

        return rpm

    @staticmethod
    def clamp(value: int, low: int, high: int) -> int:
        return max(low, min(high, value))

    # =========================================================
    # Publishers
    # =========================================================
    def publish_rpm_cmd(self, rpm_list: List[int]):
        msg = Int16MultiArray()
        msg.data = rpm_list
        self.rpm_cmd_pub.publish(msg)

        if self.publish_debug_rpm:
            self.debug_rpm_pub.publish(msg)

    def publish_brake(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.brake_pub.publish(msg)

    def publish_state(self):
        msg = String()
        msg.data = self.current_state
        self.robot_state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RobotMotorServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
