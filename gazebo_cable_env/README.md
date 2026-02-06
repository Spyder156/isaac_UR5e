# Gazebo Cable Insertion Environment

ROS2 Jazzy / gz-sim (Ignition Gazebo) simulation for cable insertion task.

## Components

- **6-DOF Arm** with Robotiq Hand-E parallel gripper
- **3 Wrist Cameras** (left, bottom, right at 90° - pointing inward)
- **Force-Torque Sensor** on wrist
- **Deformable Cable** (20 rigid segments, anchored to ground)
- **Circuit Board** with insertion socket

## Installation

```bash
# ROS2 Jazzy + gz-sim (already installed)
source /opt/ros/jazzy/setup.bash

# Build
cd ~/workspace/intrinsic_challenge
colcon build --packages-select gazebo_cable_env
source install/setup.bash
```

## Run Simulation

```bash
# Source ROS2
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Launch gz-sim simulation
ros2 launch gazebo_cable_env simulation.launch.py
```

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_states` | JointState | Arm + gripper positions |
| `/camera_left/image_raw` | Image | Left wrist camera |
| `/camera_bottom/image_raw` | Image | Bottom wrist camera |
| `/camera_right/image_raw` | Image | Right wrist camera |
| `/ft_sensor` | WrenchStamped | Force-torque at wrist |

## Control

```bash
# Move arm
ros2 topic pub /arm_controller/joint_trajectory trajectory_msgs/JointTrajectory ...

# Gripper (0=closed, 0.025=open)
ros2 topic pub /gripper_controller/commands std_msgs/Float64MultiArray "data: [0.01]"
```
