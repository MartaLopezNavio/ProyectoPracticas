# DDSM115 ROS2 controller

This project is based on the previous [Python DDSM115 library](https://github.com/rasheeddo/ddsm115_python), adapted and extended to make DDSM115 motor control easier in ROS2.

The package now supports both:
- single RS485 bus setups
- multiple Ethernet-to-RS485 converters using `socket://IP:PORT`

It also includes:
- motor ID configuration
- motor detection
- low-level velocity control
- a higher-level robot control server
- a launch file to start the full system
- a simple web monitor to visualize robot state

## Deps

- `pip3 install crcmod pyserial`

If needed, also source ROS2 before running:
```sh
source /opt/ros/humble/setup.bash
source install/setup.bash`
```

## Set Motor ID
To assign an ID to a motor, connect only one motor to the RS485 bus and run:
```sh
ros2 run ddsm115_controller set_motor_id
```
Then input the desired ID and restart the motor.
This step is only needed when configuring motors for the first time.

## Check Online Motor ID
### Single device

If you want to check motors connected to one device:
```sh
ros2 run ddsm115_controller check_motor_id
```

Or, when using Ethernet converters:
```sh
ros2 run ddsm115_controller check_motor_id --ros-args -p usb_dev:=socket://192.168.1.200:4196
```

### Multiple Ethernet converters

The node can also scan several converters at once:
```sh
ros2 run ddsm115_controller check_motor_id --ros-args -p device_urls:="['socket://192.168.1.200:4196','socket://192.168.1.201:4196','socket://192.168.1.202:4196']"
```
It will return the list of detected motor IDs for each device.

## Velocity Control

velocity_control is the low-level motor interface node. It:
- connects to one or more devices
- scans available motor IDs
- sets motors to velocity mode
- subscribes to command topics
- publishes motor feedback
This node is usually launched through the system launch file, so the operator does not need to manually pass IPs every time.

### Recommended startup
Insted of running velocity_control manually, the recommended way is: 
```sh
ros2 launch ddsm115_controller robot_system.launch.py
```
This starts: 
- velocity_control
- robot_motor_server
with the device IPs already configured.

### Manual launch
If needed, it can still be started manually: 
```sh
ros2 run ddsm115_controller velocity_control
```
Or with explicit Ethernet devices: 
```sh
ros2 run ddsm115_controller velocity_control --ros-args -p device_urls:="['socket://192.168.1.200:4196','socket://192.168.1.201:4196','socket://192.168.1.202:4196']"
```
### Velocity Control Subscriber Topics

/ddsm115/rpm_cmd    --> Type: std_msgs/msg/INt16MultiArray
This topic sends rpm commands to the motors. The array index corresponds to the motor ID: 
- index 0 --> ID 1
- index 1 --> ID 2
- etc.

For example: 
```sh
ros2 topic pub /ddsm115/rpm_cmd std_msgs/msg/Int16MultiArray "{data: [50, 0, 50]}"
```
Sends 50 rpm to ID 1, 0 to ID 2 and 50 to ID 3.

/ddsm115/brake 	   --> Type: std_msgs/msg/Bool
If True, all online motors are braked, if False, brake is released. 
For example:
```sh
ros2 topic pub /ddsm115/brake std_msgs/msg/Bool "{data: true}" -1
```

### Publisher topics

- /ddsm115/rpm_fb : as std_msgs/msg/Int16MultiArray, the node will be publishing rpm feedback of all detected motors as a list, so index 1 as ID1, index 2 as ID2, and so on. If an ID is missing in between, its index will remain 0.
- /ddsm115/cur_fb : as std_msgs/msg/Float32MultiArray, the node will be publishing current feedback in amperes unit of all detected motors. The list index follows the same convention as rpm feedback.
- /ddsm115/temp_fb : as std_msgs/msg/Int8MultiArray, the node will be publishing temperature feedback of all detected motors as a list.
- /ddsm115/error : as std_msgs/msg/Int8MultiArray, the node will be publishing error feedback of all detected motors as a list.
```sh
1 : for sensor error
2 : for over current 
4 : for phase over error
8 : for stall error
16 : for troubleshoot error
```
- /ddsm115/online_id : as std_msgs/msg/Int8MultiArray, the node will be publishing online motor ID of all detected motors as a list.

## Robot Motor Server

There is a node to make the robot easier to control from higher level commands, so the user does not need to directly handle converter IPs, motor IDs, or raw /ddsm115/rpm_cmd arrays.
```sh
ros2 run ddsm115_controller robot_motor_server
```
This node is subscribing on /cmd_vel topic and converting that to /ddsm115/rpm_cmd to drive the motors. It is also subscribing on feedback topics such as /ddsm115/rpm_fb, /ddsm115/online_id, and /ddsm115/error, then publishing a higher level robot state on /robot_state.

The node is also subscribing on /stop_robot topic, so the robot can be stopped and braked from a simple boolean command.

You can change parameters such as max_rpm, turn_gain, cmd_timeout_s, and wheel_signs to match your robot configuration. For example, wheel_signs can be used if one wheel is mounted in the opposite direction and needs inverted commands.

## Full Robot System Launch

There is a launch file to make the robot quickly start with the recommended architecture, so the user does not need to manually run each node or pass converter IPs every time.
```sh
ros2 launch ddsm115_controller robot_system.launch.py
```
This launch file starts both:
- velocity_control
- robot_motor_server
with the configured Ethernet converter addresses already included.

This is the recommended way to start the system for teleoperation, because the user only needs to interact with:
- /cmd_vel
- /stop_robot
- /robot_state
instead of dealing directly with low level motor topics.

## Web Monitor

There is also a simple monitoring node to make the robot state visible from a web browser.
```sh
ros2 run ddsm115_controller robot_web_server
```
Then open:
```sh
http://localhost:8080
```
This web page shows the robot state, online motor IDs, and the latest /cmd_vel command, so the operator can quickly check what the system is doing.

## Two Wheels Robot

There is still a node to make your robot quickly start as a mobile robot, so make sure to have left wheel motor setup as ID 1 and right wheel motor setup as ID 2.
```sh
ros2 run ddsm115_controller two_wheels_robot
## OR
ros2 run ddsm115_controller two_wheels_robot --ros-args -p pub_tf:=True
## if you need an odom->base_link TF published.
```
This node is subscribing on /cmd_vel topic then convert that to /ddsm115/rpm_cmd to drive motors, and also publishing an /odom odometry topic which calculated from /ddsm115/rpm_fb wheel's speed.

You can use rqt_robot_steering to manually drive robot OR you can use ros2 run joy joy_node to publish joystick topic to control the robot from joystick.

You can change a parameter of wheel_base to match with your robot wheel's base.

## Author

Marta López Navio

