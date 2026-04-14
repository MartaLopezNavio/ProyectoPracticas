# Mobile Robot for Vision-Based Anatomical Landmark Estimation

This repository contains the development of a mobile robotic system designed for **vision-based human pose estimation and external anatomical landmark approximation**, together with the **low-level motor control stack** used to drive the robot platform.

The project combines two main subsystems:

1. **RTMPose Mobile Vision + ROS2** 
   A vision pipeline that receives frames from a mobile phone, detects human pose using RTMPose, estimates approximate external anatomical landmarks such as neck base, thyroid, pelvis, and prostate, and publishes the results through HTTP and ROS2.

2. **DDSM115 ROS2 Controller** 
   A ROS2-based motor control package for DDSM115 hub motors, supporting both direct RS485 communication and Ethernet-to-RS485 converters. It provides low-level velocity control, motor feedback, and higher-level robot driving interfaces.

The purpose of this repository is to integrate **mobile robotics**, **computer vision**, and **ROS2 communication** into a single working system.

---

## Repository structure

```text
.
├── RTMPose/
│   ├── README.md
│   ├── mobile_pose_server.py
│   ├── landmarks_unified.py
│   ├── pose_server_discovery.py
│   ├── publish_mobile_pose_topics.py
│   ├── openmmlab_explicit.txt
│   ├── openmmlab_env.yml
│   ├── openmmlab_env_full.yml
│   └── requirements_openmmlab_pip.txt
│
└── ros2_ws/
    └── src/
        └── ddsm115_controller/
            ├── README.md
            ├── ddsm115_controller/
            ├── launch/
            ├── resource/
            ├── rviz_config/
            ├── test/
            ├── package.xml
            └── setup.py
```

---

## Main components

### 1. Vision subsystem
The vision subsystem is responsible for:

- receiving image frames from a mobile application,
- running RTMPose inference,
- computing approximate external anatomical landmarks,
- exporting results as image + JSON,
- publishing them through ROS2 topics.

More details are available in:

```text
RTMPose/README.md
```

### 2. Mobile base / motor control subsystem
The motor control subsystem is responsible for:

- communicating with DDSM115 motors,
- detecting online motor IDs,
- sending wheel velocity commands,
- publishing motor feedback,
- exposing higher-level robot control interfaces through ROS2.

More details are available in:

```text
ros2_ws/src/ddsm115_controller/README.md
```

---

## High-level workflow

A typical workflow is:

1. Start the motor control stack for the mobile base.
2. Start the RTMPose vision server and ROS2 publisher.
3. Send frames from the mobile application to the vision server.
4. Read vision outputs and ROS2 topics for landmark estimation.
5. Use the robot platform and the vision outputs as part of the complete experimental setup.

---

## Technologies used

- ROS2 Humble
- Python 3.8
- PyTorch
- RTMPose / MMPose
- MMCV / MMEngine / MMDetection
- Flask
- OpenCV
- DDSM115 motor drivers
- RS485 / Ethernet-to-RS485 communication

---

## How to use this repository

This repository contains multiple subsystems. 
Please refer to the corresponding README depending on what you want to run:

- **Vision pipeline:** `RTMPose/README.md`
- **Motor controller:** `ros2_ws/src/ddsm115_controller/README.md`

---

## Project status

This repository is intended as a development and integration workspace for an ongoing robotics and computer vision project. Some components are experimental and may evolve over time.

---

## Notes

The anatomical landmarks estimated by the vision subsystem are **external approximations based on 2D pose estimation** and should not be interpreted as exact medical localization.

---

## Author

Marta López Navio
