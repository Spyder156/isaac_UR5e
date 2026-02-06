#!/usr/bin/env python3
"""
Launch file for Cable Insertion Simulation (gz-sim / Gazebo Harmonic)
Spawns UR5e + Hand-E + Cable + Circuit Board with ros2_control
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    TimerAction
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_path = get_package_share_directory('gazebo_cable_env')

    # Xacro file
    xacro_file = os.path.join(pkg_path, 'urdf', 'ur5e_hande.urdf.xacro')

    # World file (SDF for gz-sim)
    world_file = os.path.join(pkg_path, 'worlds', 'cable_insertion.sdf')

    # Robot description from xacro
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=str
    )

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Start gz-sim (Gazebo Harmonic)
    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_file],
        output='screen'
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time
        }]
    )

    # Spawn robot in gz-sim using ros_gz_sim
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_robot',
        output='screen',
        arguments=[
            '-topic', '/robot_description',
            '-name', 'ur5e_hande',
            '-z', '0.0'
        ]
    )

    # Joint State Broadcaster Spawner
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    # Arm Controller Spawner
    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    # Gripper Controller Spawner
    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    # Bridge gz topics to ROS2
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        arguments=[
            # Camera topics (gz-sim format)
            '/camera_left/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_bottom/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_right/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            # Force-Torque sensor
            '/ft_sensor@geometry_msgs/msg/WrenchStamped[gz.msgs.Wrench',
            # Clock
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        output='screen'
    )

    # Delay spawning until robot_state_publisher is up
    delayed_spawn = TimerAction(
        period=2.0,
        actions=[spawn_robot]
    )

    # Spawn controllers after robot is spawned (chain events with delays)
    delayed_joint_state_broadcaster = TimerAction(
        period=5.0,  # Wait for robot to be fully loaded
        actions=[joint_state_broadcaster_spawner]
    )

    # Spawn arm controller after joint_state_broadcaster
    delayed_arm_controller = TimerAction(
        period=7.0,
        actions=[arm_controller_spawner]
    )

    # Spawn gripper controller after arm controller
    delayed_gripper_controller = TimerAction(
        period=8.0,
        actions=[gripper_controller_spawner]
    )

    return LaunchDescription([
        # Declare arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),

        # Start gz-sim
        gz_sim,

        # Robot State Publisher
        robot_state_publisher,

        # Spawn robot (delayed)
        delayed_spawn,

        # Bridge topics
        gz_bridge,

        # Spawn controllers (delayed and chained)
        delayed_joint_state_broadcaster,
        delayed_arm_controller,
        delayed_gripper_controller,
    ])
