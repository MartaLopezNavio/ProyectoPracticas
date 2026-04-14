from setuptools import setup
import os
from glob import glob

package_name = 'ddsm115_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Marta López',
    maintainer_email='al417883@uji.es',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'check_motor_id = ddsm115_controller.check_motor_id:main',
            'set_motor_id = ddsm115_controller.set_motor_id:main',
            'velocity_control = ddsm115_controller.velocity_control:main',
            'two_wheels_robot = ddsm115_controller.two_wheels_robot:main',
            'robot_motor_server = ddsm115_controller.robot_motor_server:main',
            'robot_web_server = ddsm115_controller.robot_web_server:main',
        ],
    },
)
