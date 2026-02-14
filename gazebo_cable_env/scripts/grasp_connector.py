#!/usr/bin/env python3
"""
Script to move robot arm to connector and close gripper to grasp it.
Uses TF to find the actual connector position.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_ros import TransformListener, Buffer
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration as DurationMsg
import time
import math


class GraspConnector(Node):
    def __init__(self):
        super().__init__('grasp_connector')

        # Set use_sim_time parameter
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # QoS for reliable communication
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Publishers for arm and gripper controllers
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            qos
        )
        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            '/gripper_controller/joint_trajectory',
            qos
        )

        # Subscribe to joint states to verify commands
        self.joint_states = None
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.get_logger().info('Grasp connector node started (use_sim_time=True)')

    def joint_state_callback(self, msg):
        self.joint_states = msg

    def spin_for(self, seconds):
        """Spin the node for a given number of seconds."""
        start = time.time()
        while time.time() - start < seconds:
            rclpy.spin_once(self, timeout_sec=0.1)

    def list_available_frames(self):
        """List all available TF frames."""
        try:
            frames = self.tf_buffer.all_frames_as_yaml()
            if frames:
                self.get_logger().info(f'Available TF frames:\n{frames[:500]}...')  # Truncate
            else:
                self.get_logger().warn('No TF frames available yet')
        except Exception as e:
            self.get_logger().warn(f'Could not list frames: {e}')

    def get_connector_position(self):
        """Get connector position from TF."""
        base_frames = ['base_link', 'base', 'world']
        target_frame = 'connector_grip'

        self.list_available_frames()

        for base_frame in base_frames:
            try:
                if self.tf_buffer.can_transform(base_frame, target_frame, rclpy.time.Time()):
                    transform = self.tf_buffer.lookup_transform(
                        base_frame,
                        target_frame,
                        rclpy.time.Time()
                    )
                    pos = transform.transform.translation
                    self.get_logger().info(f'Found connector at ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}) in {base_frame}')

                    if base_frame == 'base_link':
                        return (pos.x, pos.y, pos.z + 0.425)
                    return (pos.x, pos.y, pos.z)
            except Exception as e:
                self.get_logger().debug(f'Frame {base_frame} lookup failed: {e}')

        # Fallback based on where the cable droops to
        self.get_logger().warn('TF lookup failed, using estimated position')
        # Cable anchor at (0.25, 0.25, 0.435), cable sags - connector should be near table
        return (0.30, 0.30, 0.50)

    def open_gripper(self):
        """Open the gripper."""
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = ['hande_left_finger_joint', 'hande_right_finger_joint']

        point = JointTrajectoryPoint()
        point.positions = [0.025, 0.025]  # Fully open
        point.velocities = [0.0, 0.0]
        point.time_from_start = DurationMsg(sec=1, nanosec=0)

        msg.points = [point]

        self.get_logger().info('Opening gripper...')
        for _ in range(3):  # Publish multiple times to ensure delivery
            self.gripper_pub.publish(msg)
            self.spin_for(0.1)

    def close_gripper(self):
        """Close the gripper to grasp connector."""
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = ['hande_left_finger_joint', 'hande_right_finger_joint']

        point = JointTrajectoryPoint()
        point.positions = [0.005, 0.005]  # Close to grip
        point.velocities = [0.0, 0.0]
        point.time_from_start = DurationMsg(sec=1, nanosec=0)

        msg.points = [point]

        self.get_logger().info('Closing gripper...')
        for _ in range(3):
            self.gripper_pub.publish(msg)
            self.spin_for(0.1)

    def move_arm(self, positions, duration_sec=3):
        """Move arm to specified joint positions."""
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0] * 6
        point.time_from_start = DurationMsg(sec=duration_sec, nanosec=0)

        msg.points = [point]

        self.get_logger().info(f'Moving arm: {[f"{p:.2f}" for p in positions]}')
        for _ in range(3):
            self.arm_pub.publish(msg)
            self.spin_for(0.1)

    def move_to_connector(self, target_x, target_y, target_z):
        """Calculate joint angles to reach target position."""
        # shoulder_pan to face target
        shoulder_pan = math.atan2(target_y, target_x)

        # Adjust based on height
        if target_z < 0.55:
            shoulder_lift = -1.6
            elbow = 1.3
            wrist_1 = -1.2
        else:
            shoulder_lift = -1.3
            elbow = 1.0
            wrist_1 = -1.3

        wrist_2 = -1.57
        wrist_3 = shoulder_pan

        self.get_logger().info(f'Target: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})')
        self.move_arm([shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3])


def main():
    rclpy.init()
    node = GraspConnector()

    # Spin to receive TF data and joint states
    node.get_logger().info('Waiting for simulation data (10 seconds)...')
    node.spin_for(10.0)

    # Check if we're receiving joint states
    if node.joint_states:
        node.get_logger().info(f'Receiving joint states: {node.joint_states.name[:3]}...')
    else:
        node.get_logger().warn('No joint states received!')

    # Get connector position
    connector_pos = node.get_connector_position()

    node.get_logger().info('=== Starting grasp sequence ===')

    # 1. Open gripper
    node.open_gripper()
    node.spin_for(2.0)

    # 2. Move to connector
    node.move_to_connector(*connector_pos)
    node.spin_for(4.0)

    # 3. Close gripper
    node.close_gripper()
    node.spin_for(2.0)

    node.get_logger().info('=== Grasp sequence complete ===')

    node.spin_for(1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
