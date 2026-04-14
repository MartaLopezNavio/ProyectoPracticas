'''
    This code is to interface with DDSM115 python driver.
    Supports one or multiple transport endpoints (USB or socket://IP:PORT).
    It scans all configured devices, builds an ID->driver map, and keeps the
    same ROS topics:
      /ddsm115/rpm_cmd
      /ddsm115/brake
      /ddsm115/rpm_fb
      /ddsm115/cur_fb
      /ddsm115/temp_fb
      /ddsm115/error
      /ddsm115/online_id
'''

import rclpy
from rclpy.node import Node
from ddsm115_controller.ddsm115 import *
from std_msgs.msg import Int16MultiArray, Int8MultiArray, Float32MultiArray, Bool
import time


class VelocityControl(Node):

    def __init__(self):
        super().__init__("velocity_control_node")
        self.get_logger().info('Start velocity_control_node')

        self.declare_parameter('max_check', 10)
        self.declare_parameter('usb_dev', "/dev/ttyUSB0")
        # STRING_ARRAY en Humble
        self.declare_parameter('device_urls', [''])

        self.max_check = self.get_parameter('max_check').get_parameter_value().integer_value
        self.usb_dev = self.get_parameter('usb_dev').get_parameter_value().string_value
        self.device_urls = [x for x in list(self.get_parameter('device_urls').value) if x]

        self.get_logger().info("Using parameters as below")
        self.get_logger().info("max_check: {}".format(self.max_check))
        self.get_logger().info("usb_dev: {}".format(self.usb_dev))
        self.get_logger().info("device_urls: {}".format(self.device_urls))

        # Si no pasan lista, comportamiento antiguo
        if not self.device_urls:
            self.device_urls = [self.usb_dev]

        # device string -> MotorControl
        self.drivers = {}
        # motor_id -> MotorControl
        self.id_to_driver = {}
        # motor_id -> device string
        self.id_to_device = {}
        self.online_id = []

        ### Check available motors on all devices ###
        for dev in self.device_urls:
            try:
                driver = MotorControl(dev)
                self.drivers[dev] = driver
                self.get_logger().info(f"Scanning device {dev}")

                for i in range(self.max_check):
                    motor_id = i + 1
                    data_fb = driver.get_motor_feedback(motor_id)
                    if data_fb['id'] is not None:
                        found_id = int(data_fb['id'])

                        if found_id in self.id_to_driver:
                            self.get_logger().warn(
                                f"Duplicate motor ID {found_id} detected on {dev}. "
                                f"Already registered on {self.id_to_device[found_id]}. Ignoring duplicate."
                            )
                            continue

                        self.id_to_driver[found_id] = driver
                        self.id_to_device[found_id] = dev
                        self.online_id.append(found_id)

            except Exception as e:
                self.get_logger().error(f"Failed to open {dev}: {e}")

        self.online_id = sorted(self.online_id)
        self.get_logger().info("Online ID is {}".format(self.online_id))

        self.total_motor = len(self.online_id)

        if self.total_motor == 0:
            self.get_logger().info("No motor detected...")
            quit()

        self.last_motor_id = max(self.online_id)

        # Set all detected motors in velocity mode
        for motor_id in self.online_id:
            try:
                ret = self.id_to_driver[motor_id].set_drive_mode(_id=motor_id, _mode=2)
                self.get_logger().info("{} on {}".format(ret, self.id_to_device[motor_id]))
            except Exception as e:
                self.get_logger().error(f"Failed to set velocity mode for ID {motor_id}: {e}")

        ### Variables ###
        self.rpm_cmd_list = [None] * self.last_motor_id
        self.last_rpm_recv_stamp = time.time()
        self.brake_enable = False
        self.last_slow_pub_stamp = time.time()

        self.rpm_fb_list = [0] * self.last_motor_id
        self.temp_fb_list = [0] * self.last_motor_id
        self.cur_fb_list = [0.0] * self.last_motor_id
        self.error_list = [0] * self.last_motor_id

        ### Pub/Sub ###
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.rpm_cmd_sub = self.create_subscription(
            Int16MultiArray, "/ddsm115/rpm_cmd", self.rpm_cmd_callback, qos_profile=qos
        )
        self.brake_cmd_sub = self.create_subscription(
            Bool, "/ddsm115/brake", self.brake_cmd_callback, qos_profile=qos
        )

        self.rpm_fb_pub = self.create_publisher(Int16MultiArray, "/ddsm115/rpm_fb", qos_profile=qos)
        self.cur_fb_pub = self.create_publisher(Float32MultiArray, "/ddsm115/cur_fb", qos_profile=qos)
        self.temp_fb_pub = self.create_publisher(Int8MultiArray, "/ddsm115/temp_fb", qos_profile=qos)
        self.error_pub = self.create_publisher(Int8MultiArray, "/ddsm115/error", qos_profile=qos)
        self.motor_online_pub = self.create_publisher(Int8MultiArray, "/ddsm115/online_id", qos_profile=qos)

        self.get_logger().info('----------------------------------- Publishers --------------------------------------')
        self.get_logger().info('Publish motors rpm feedback to      /ddsm115/rpm_fb     [std_msgs/msg/Int16MultiArray]')
        self.get_logger().info('Publish motors current feedback to  /ddsm115/cur_fb     [std_msgs/msg/Float32MultiArray]')
        self.get_logger().info('Publish motors temp feedback to     /ddsm115/temp_fb    [std_msgs/msg/Int8MultiArray]')
        self.get_logger().info('Publish motors error feedback to    /ddsm115/error      [std_msgs/msg/Int8MultiArray]')
        self.get_logger().info('Publish motors online id to         /ddsm115/online_id  [std_msgs/msg/Int8MultiArray]')
        self.get_logger().info('---------------------------------- Subscribers --------------------------------------')
        self.get_logger().info('Subscribe motors rpm cmd at         /ddsm115/rpm_cmd    [std_msgs/msg/Int16MultiArray]')
        self.get_logger().info('Subscribe motors brake cmd at       /ddsm115/brake      [std_msgs/msg/Bool]')

        self.timer = self.create_timer(0.01, self.timer_callback)

    #####################
    ### ROS callbacks ###
    #####################
    def rpm_cmd_callback(self, msg):
        for i, data in enumerate(msg.data):
            if i < len(self.rpm_cmd_list):
                self.rpm_cmd_list[i] = data

        self.last_rpm_recv_stamp = time.time()

    def brake_cmd_callback(self, msg):
        self.get_logger().info("Got brake {}".format(msg.data))
        self.brake_enable = bool(msg.data)

    ##############
    ### Helper ###
    ##############
    def set_rpm(self):
        self.brake_enable = False

        for motor_id in self.online_id:
            idx = motor_id - 1
            rpm_cmd = self.rpm_cmd_list[idx]

            if rpm_cmd is not None:
                try:
                    self.id_to_driver[motor_id].send_rpm(motor_id, rpm_cmd)
                except Exception as e:
                    self.get_logger().warn(f"send_rpm failed for ID {motor_id}: {e}")

    def set_zero_rpm(self):
        for motor_id in self.online_id:
            self.rpm_cmd_list[motor_id - 1] = 0
        self.set_rpm()

    def brake_motors(self):
        for motor_id in self.online_id:
            try:
                self.id_to_driver[motor_id].set_brake(_id=motor_id)
            except Exception as e:
                self.get_logger().warn(f"set_brake failed for ID {motor_id}: {e}")

    ############
    ### Loop ###
    ############
    def timer_callback(self):
        if (time.time() - self.last_rpm_recv_stamp) > 2.0:
            if not self.brake_enable:
                self.set_zero_rpm()
            else:
                self.brake_motors()
        else:
            self.set_rpm()

        ## Feedback ##
        for motor_id in self.online_id:
            try:
                data_fb = self.id_to_driver[motor_id].get_motor_feedback(_id=motor_id)
                if data_fb['id'] is not None:
                    self.rpm_fb_list[motor_id - 1] = data_fb['fb_rpm']
                    self.temp_fb_list[motor_id - 1] = data_fb['wind_temp']
                    self.cur_fb_list[motor_id - 1] = data_fb['fb_cur']
                    self.error_list[motor_id - 1] = data_fb['error']
            except Exception as e:
                self.get_logger().warn(f"Feedback failed for ID {motor_id}: {e}")

        rpm_msg = Int16MultiArray()
        rpm_msg.data = self.rpm_fb_list
        self.rpm_fb_pub.publish(rpm_msg)

        cur_msg = Float32MultiArray()
        cur_msg.data = self.cur_fb_list
        self.cur_fb_pub.publish(cur_msg)

        if (time.time() - self.last_slow_pub_stamp) > 0.1:
            temp_msg = Int8MultiArray()
            temp_msg.data = self.temp_fb_list
            self.temp_fb_pub.publish(temp_msg)

            error_msg = Int8MultiArray()
            error_msg.data = self.error_list
            self.error_pub.publish(error_msg)

            online_id_msg = Int8MultiArray()
            online_id_msg.data = self.online_id
            self.motor_online_pub.publish(online_id_msg)

            self.last_slow_pub_stamp = time.time()

    def destroy(self):
        for dev, driver in self.drivers.items():
            try:
                driver.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VelocityControl()
    rclpy.spin(node)
    node.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
