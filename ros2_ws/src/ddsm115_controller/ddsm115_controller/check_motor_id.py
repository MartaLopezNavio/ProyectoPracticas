import rclpy
from rclpy.node import Node
from ddsm115_controller.ddsm115 import *

class CheckMotorId(Node):

    def __init__(self):
        super().__init__("check_motor_id_node")
        self.get_logger().info('Start check_motor_id_node')

        self.declare_parameter('max_check', 10)
        self.declare_parameter('usb_dev', "/dev/ttyUSB0")
        # CAMBIO: usar [''] para forzar STRING_ARRAY en ROS2
        self.declare_parameter('device_urls', [''])

        self.max_check = self.get_parameter('max_check').get_parameter_value().integer_value
        self.usb_dev = self.get_parameter('usb_dev').get_parameter_value().string_value
        # CAMBIO: quitar strings vacíos
        self.device_urls = [x for x in list(self.get_parameter('device_urls').value) if x]

        self.get_logger().info("Using parameters as below")
        self.get_logger().info("max_check: {}".format(self.max_check))
        self.get_logger().info("usb_dev: {}".format(self.usb_dev))
        self.get_logger().info("device_urls: {}".format(self.device_urls))

        if not self.device_urls:
            self.device_urls = [self.usb_dev]

        for dev in self.device_urls:
            self.get_logger().info(f"Checking device: {dev}")
            try:
                d = MotorControl(dev)
                online_id = []

                for i in range(self.max_check):
                    data_fb = d.get_motor_feedback(i + 1)
                    if data_fb['id'] is not None:
                        online_id.append(data_fb['id'])

                self.get_logger().info(f"[{dev}] Online ID is {online_id}")

                try:
                    d.close()
                except Exception:
                    pass

            except Exception as e:
                self.get_logger().error(f"[{dev}] Check failed: {e}")

        quit()


def main(args=None):
    rclpy.init(args=args)
    node = CheckMotorId()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main() 
