'''
	This code is designed to use with velocity_control.py node.
	It assumes the left motor is ID1 as first index, and 
	right motor is ID2 as second index of both /ddsm115/rpm_fb & rpm_cmd.

	
'''

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, Int8MultiArray, Float32MultiArray, Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
import time
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class TwoWheelsRobot(Node):

	def __init__(self):
		super().__init__("two_wheels_robot_node")
		self.get_logger().info('Start two_wheels_robot_node')

		### ROS Parameters ###
		self.declare_parameter("wheel_base", 0.255)
		self.declare_parameter("R_wheel", 0.051)
		self.declare_parameter("pub_tf", False)
		
		self.wheel_base = self.get_parameter('wheel_base').get_parameter_value().double_value
		self.R_wheel = self.get_parameter('R_wheel').get_parameter_value().double_value
		self.pub_tf = self.get_parameter('pub_tf').get_parameter_value().bool_value

		self.get_logger().info("Using parameters as below")
		self.get_logger().info("wheel_base: {}".format(self.wheel_base))
		self.get_logger().info("R_wheel: {}".format(self.R_wheel))
		self.get_logger().info("pub_tf: {}".format(self.pub_tf))

		### Variables ###
		self.rpm_fb_left = 0
		self.rpm_fb_right = 0
		self.x = 0.0
		self.y = 0.0
		self.theta = 0.0
		self.period = 0.01
		self.prev_y = 0.0

		self.thr_stick = 0.0
		self.str_stick = 0.0
		self.cart_mode = 2
		self.prev_cart_mode = self.cart_mode

		self.max_rpm = 150

		### TF ###
		self.br = TransformBroadcaster(self)

		### Pub/Sub ###
		qos = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT, \
											history=rclpy.qos.HistoryPolicy.KEEP_LAST, \
											depth=1)
		self.rpm_cmd_pub = self.create_publisher(Int16MultiArray, "/ddsm115/rpm_cmd", qos_profile=qos)
		self.odom_pub = self.create_publisher(Odometry, "/odom", 10)

		self.rpm_fb_sub = self.create_subscription(Int16MultiArray, "/ddsm115/rpm_fb", self.rpm_fb_callback, qos_profile=qos)
		self.cmd_vel_sub = self.create_subscription(Twist, "/cmd_vel", self.cmd_vel_callback, 10)
		self.joy_sub = self.create_subscription(Joy, "/joy", self.joy_callback, 10)

		self.timer = self.create_timer(self.period, self.timer_callback)

	def _print(self, msg):
		self.get_logger().info("{}".format(msg))

	#####################
	### ROS Callbacks ###
	#####################
	def rpm_fb_callback(self, msg):
		self.rpm_fb_left = msg.data[0]
		self.rpm_fb_right = -msg.data[1]

	def cmd_vel_callback(self, msg):
		vx = msg.linear.x
		wz = msg.angular.z

		if (vx != 0.0) and (wz == 0.0):
			vl = vx
			vr = vx

		elif (vx == 0.0) and (wz != 0.0):

			vl = -wz * self.wheel_base/2.0
			vr = wz * self.wheel_base/2.0

		elif (vx != 0.0) and (wz != 0.0):
			R_icc = abs(vx)/abs(wz)
			sign_vx = vx/abs(vx)
			if wz > 0.0:
				# print("curve left")
				vl = (sign_vx)*(wz*(R_icc - self.wheel_base/2.0)) #/2.0
				vr = (sign_vx)*(wz*(R_icc + self.wheel_base/2.0)) #/2.0
			elif wz < 0.0:
				# print("curve right")
				vl = (sign_vx)*(abs(wz)*(R_icc + self.wheel_base/2.0)) #/2.0
				vr = (sign_vx)*(abs(wz)*(R_icc - self.wheel_base/2.0)) #/2.0
		else:
			vl = 0.0
			vr = 0.0


		left_rpm = int(self.linear_to_rpm(vl))
		right_rpm = int(self.linear_to_rpm(vr)*(-1.0))

		rpm_cmd_msg = Int16MultiArray()
		rpm_cmd_msg.data = [left_rpm, right_rpm]

		self.rpm_cmd_pub.publish(rpm_cmd_msg)
		self._print("Cmd_vel vx: {:.2f} wz: {:.2f} vl: {:.2f} vr: {:.2f} rpmL: {:d} rpmR: {:d}".format(\
			vx, wz, vl, vr, left_rpm, right_rpm))
		
	def joy_callback(self, msg):

		self.thr_stick = msg.axes[1]*100.0
		self.str_stick = -msg.axes[3]*100.0

		## A button AUTO
		if (msg.buttons[0] == 1):
			self.prev_cart_mode = self.cart_mode
			self.cart_mode = 2
			self._print("cart mode is 2")

		## X button MANUAL
		elif (msg.buttons[2] == 1):
			self.prev_cart_mode = self.cart_mode
			self.cart_mode = 1
			self._print("cart mode is 1")

	######################
	### Math functions ###
	######################
	def map_with_limit(self, val, in_min, in_max, out_min, out_max):

		m = (out_max - out_min)/(in_max - in_min)
		out = m*(val - in_min) + out_min

		if out_min > out_max:
			if out > out_min:
				out = out_min
			elif out < out_max:
				out = out_max
			else:
				pass
		elif out_max > out_min:
			if out > out_max:
				out = out_max
			elif out < out_min:
				out = out_min
			else:
				pass
		else:
			pass

		return out

	def linear_to_rpm(self, v):
		return (60.0/(2.0*np.pi))*(v/self.R_wheel)

	def rpm_to_linear(self, rpm):
		return rpm * ((2.0*np.pi)/60.0) * self.R_wheel
	
	def xy_mixing(self, x, y):
		## x, y must be in the range of -100 to 100

		left = y+x
		right = y-x

		diff = abs(x) - abs(y)

		if (left < 0.0):
			left = left - abs(diff)
		else:
			left = left + abs(diff)

		if (right < 0.0):
			right = right - abs(diff)
		else:
			right = right + abs(diff)

		if (self.prev_y < 0.0):
			swap = left
			left = right
			right = swap
		
		self.prev_y = y

		## left and right are in -200 to 200 ranges

		return left, right


	############
	### Loop ###
	############	
	def timer_callback(self):

		if self.cart_mode == 1:
			if (-5.0 <= self.thr_stick <= 5.0) and (-5.0 <= self.str_stick <= 5.0):
				left_rpm = 0
				right_rpm = 0
			else:
				left_200_per, right_200_per = self.xy_mixing(self.str_stick, self.thr_stick)
				left_rpm = int(self.map_with_limit(left_200_per, -200.0, 200.0, -self.max_rpm, self.max_rpm))
				right_rpm = int(self.map_with_limit(right_200_per, -200.0, 200.0, self.max_rpm, -self.max_rpm))

			rpm_cmd_msg = Int16MultiArray()
			rpm_cmd_msg.data = [left_rpm, right_rpm]
			self.rpm_cmd_pub.publish(rpm_cmd_msg)

			self._print("Joystick str_stick: {:.2f} thr_stick: {:.2f} rpmL: {:d} rpmR: {:d}".format(\
				self.str_stick, self.thr_stick, left_rpm, right_rpm))



		### Calculate ODOM ###
		vl = round(self.rpm_to_linear(self.rpm_fb_left), 3)
		vr = round(self.rpm_to_linear(self.rpm_fb_right), 3)

		V = (vl + vr)/2.0

		if (vl > 0.0) and (vr < 0.0) and (abs(V) < 0.1):
			## rotatiing CW
			V = 0.0
			Wz = (vr - vl)/self.wheel_base
			self.theta = self.theta + Wz*self.period

			path = "skid_right"

		elif (vr > 0.0) and (vl < 0.0) and (abs(V) < 0.1):
			## rotatiing CCW
			V = 0.0
			Wz = (vr - vl)/self.wheel_base
			self.theta = self.theta + Wz*self.period

			path = "skid_left"

		elif (abs(vl) > abs(vr)) or (abs(vl) < abs(vr)):
			## curving CW
			# V = (vl + vr)/2.0
			Wz = (vr-vl)/self.wheel_base
			# R_ICC = (self.L/2.0)*((vl+vr)/(vl-vr))
			R_ICC = (self.wheel_base/2.0)*((vl+vr)/(vr-vl))

			self.x = self.x - R_ICC*np.sin(self.theta) + R_ICC*np.sin(self.theta + Wz*self.period)
			self.y = self.y + R_ICC*np.cos(self.theta) - R_ICC*np.cos(self.theta + Wz*self.period)
			self.theta = self.theta + Wz*self.period

			if abs(vl) > abs(vr):
				path = "curve_right"
			else:
				path = "curve_left"

		elif vl == vr:
			V = (vl + vr)/2.0
			Wz = 0.0
			self.x = self.x + V*np.cos(self.theta)*self.period
			self.y = self.y + V*np.sin(self.theta)*self.period
			self.theta = self.theta
			path = "straight"

		else:
			V = 0.0
			Wz = 0.0
			R_ICC = 0.0

		q = quaternion_from_euler(0,0, self.theta)
		odom_msg = Odometry()
		odom_msg.header.stamp = self.get_clock().now().to_msg()
		odom_msg.header.frame_id = "odom"
		odom_msg.child_frame_id = "base_link"	
		odom_msg.pose.pose.position.x = self.x
		odom_msg.pose.pose.position.y = self.y
		odom_msg.pose.pose.position.z = 0.0
		odom_msg.pose.pose.orientation.x = q[0]
		odom_msg.pose.pose.orientation.y = q[1]
		odom_msg.pose.pose.orientation.z = q[2]
		odom_msg.pose.pose.orientation.w = q[3]
		odom_msg.pose.covariance[0] = 0.0001
		odom_msg.pose.covariance[7] = 0.0001
		odom_msg.pose.covariance[14] = 0.000001	#1e12
		odom_msg.pose.covariance[21] = 0.000001	#1e12
		odom_msg.pose.covariance[28] = 0.000001	#1e12
		odom_msg.pose.covariance[35] = 0.0001
		odom_msg.twist.twist.linear.x = V 
		odom_msg.twist.twist.linear.y = 0.0 
		odom_msg.twist.twist.angular.z = Wz 
		self.odom_pub.publish(odom_msg)

		if self.pub_tf:
			# construct tf
			t = TransformStamped()
			t.header.frame_id = "odom" 
			t.header.stamp = self.get_clock().now().to_msg()
			t.child_frame_id = "base_link"	
			t.transform.translation.x = self.x 
			t.transform.translation.y = self.y
			t.transform.translation.z = 0.0

			t.transform.rotation.x = odom_msg.pose.pose.orientation.x
			t.transform.rotation.y = odom_msg.pose.pose.orientation.y
			t.transform.rotation.z = odom_msg.pose.pose.orientation.z
			t.transform.rotation.w = odom_msg.pose.pose.orientation.w
			self.br.sendTransform(t)

def main(args=None):
	rclpy.init(args=args)
	node = TwoWheelsRobot()
	rclpy.spin(node)
	node.destroy()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
