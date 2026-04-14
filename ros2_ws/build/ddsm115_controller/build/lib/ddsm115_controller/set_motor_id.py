'''
	This code is to set motor ID.
	ATTENTION! Please plug only one motor at a time.

	Developed by Rasheed Kittinanthapanya

	https://github.com/rasheeddo/
	https://www.youtube.com/@stepbystep-robotics
'''

import rclpy
from rclpy.node import Node
from ddsm115_controller.ddsm115 import *

class SetMotorId(Node):

	def __init__(self):
		super().__init__("set_motor_id_node")
		self.get_logger().info('Start set_motor_id_node')

		self.declare_parameter('usb_dev', "/dev/ttyUSB0")

		self.usb_dev = self.get_parameter('usb_dev').get_parameter_value().string_value

		self.get_logger().info("Using parameters as below")
		self.get_logger().info("usb_dev: {}".format(self.usb_dev))

		d = MotorControl(self.usb_dev)
		self.get_logger().info("Make sure to have only one motor plugged on")
		_id = input("Once ready input ID you want to set, then press [Enter] ")
		while not isinstance(int(_id), int):
			_id = input("Input ID is not integer! input again, then press [Enter] ")

		d.set_id(int(_id))
		self.get_logger().info("Motor ID is set to {}".format(_id))
		self.get_logger().info("Please restart the motor again.")

		quit()

def main(args=None):
	rclpy.init(args=args)
	node = SetMotorId()
	rclpy.spin(node)
	node.destroy()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
