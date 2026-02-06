#!/usr/bin/env python3
"""
Test script to move the UR5e robot and verify cable physics.
This script sends joint trajectory commands to make the robot move
so you can observe the cable deformation.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Duration
import math
import time


class RobotMover(Node):
    def __init__(self):
        super().__init__('robot_mover')

        # Action client for arm trajectory control
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )

        # Publisher for gripper control
        self.gripper_pub = self.create_publisher(
            Float64MultiArray,
            '/gripper_controller/commands',
            10
        )

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        self.get_logger().info('Waiting for arm controller...')
        self.arm_client.wait_for_server()
        self.get_logger().info('Arm controller connected!')

    def send_arm_trajectory(self, positions_list, durations):
        """Send a trajectory with multiple waypoints."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names

        for positions, duration in zip(positions_list, durations):
            point = JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
            goal.trajectory.points.append(point)

        self.get_logger().info(f'Sending trajectory with {len(positions_list)} waypoints...')
        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected!')
            return False

        self.get_logger().info('Goal accepted, executing...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return True

    def set_gripper(self, position):
        """Set gripper position (0 = closed, 0.025 = open)."""
        msg = Float64MultiArray()
        msg.data = [position, position]  # Both fingers
        self.gripper_pub.publish(msg)
        self.get_logger().info(f'Gripper set to {position}')

    def run_test_sequence(self):
        """Run a test sequence to move the robot and observe cable physics."""

        self.get_logger().info('=' * 50)
        self.get_logger().info('Starting robot movement test')
        self.get_logger().info('Watch the cable to verify physics!')
        self.get_logger().info('=' * 50)

        # Home position (arm up)
        home = [0.0, -math.pi/2, 0.0, -math.pi/2, 0.0, 0.0]

        # Position 1: Reach forward and down
        pos1 = [0.0, -math.pi/3, math.pi/3, -math.pi/2, -math.pi/2, 0.0]

        # Position 2: Sweep to the side (this should move cable)
        pos2 = [math.pi/4, -math.pi/3, math.pi/3, -math.pi/2, -math.pi/2, 0.0]

        # Position 3: Sweep to other side
        pos3 = [-math.pi/4, -math.pi/3, math.pi/3, -math.pi/2, -math.pi/2, 0.0]

        # Position 4: Reach toward cable anchor area
        pos4 = [math.pi/2, -math.pi/4, math.pi/4, -math.pi/2, -math.pi/2, 0.0]

        # Position 5: Reach toward socket board
        pos5 = [-math.pi/4, -math.pi/4, math.pi/3, -math.pi/3, -math.pi/2, 0.0]

        self.get_logger().info('\n>>> Moving to home position...')
        self.send_arm_trajectory([home], [3.0])
        time.sleep(1)

        self.get_logger().info('\n>>> Opening gripper...')
        self.set_gripper(0.025)
        time.sleep(1)

        self.get_logger().info('\n>>> Moving forward and down (pos1)...')
        self.send_arm_trajectory([pos1], [3.0])
        time.sleep(1)

        self.get_logger().info('\n>>> Sweeping to right side (pos2)...')
        self.send_arm_trajectory([pos2], [2.0])
        time.sleep(1)

        self.get_logger().info('\n>>> Sweeping to left side (pos3)...')
        self.send_arm_trajectory([pos3], [3.0])
        time.sleep(1)

        self.get_logger().info('\n>>> Reaching toward cable area (pos4)...')
        self.send_arm_trajectory([pos4], [3.0])
        time.sleep(1)

        self.get_logger().info('\n>>> Closing gripper (simulating grab)...')
        self.set_gripper(0.005)
        time.sleep(1)

        self.get_logger().info('\n>>> Moving toward socket (pos5)...')
        self.send_arm_trajectory([pos5], [3.0])
        time.sleep(1)

        self.get_logger().info('\n>>> Returning to home...')
        self.send_arm_trajectory([home], [4.0])
        time.sleep(1)

        self.get_logger().info('\n>>> Opening gripper...')
        self.set_gripper(0.025)

        self.get_logger().info('=' * 50)
        self.get_logger().info('Test sequence complete!')
        self.get_logger().info('=' * 50)


def main():
    rclpy.init()

    mover = RobotMover()

    try:
        mover.run_test_sequence()
    except KeyboardInterrupt:
        pass
    finally:
        mover.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
