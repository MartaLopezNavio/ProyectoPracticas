from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ddsm115_controller',
            executable='velocity_control',
            name='velocity_control_node',
            output='screen',
            parameters=[{
                'max_check': 10,
                'device_urls': [
                    'socket://192.168.1.200:4196',
                    'socket://192.168.1.201:4196',
                    'socket://192.168.1.202:4196',
                ],
            }]
        ),
        Node(
            package='ddsm115_controller',
            executable='robot_motor_server',
            name='robot_motor_server',
            output='screen',
            parameters=[{
                'num_wheels': 3,
                'max_rpm': 200,
                'turn_gain': 100.0,
                'cmd_timeout_s': 1.0,
                'publish_debug_rpm': True,
                'wheel_signs': [-1, 1, 1],
            }]
        ),
    ])
