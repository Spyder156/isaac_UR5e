#!/usr/bin/env python3
"""
Test script for Cable Insertion Gazebo Environment
Checks that all expected topics are published
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import WrenchStamped
import sys


class SimulationTester(Node):
    def __init__(self):
        super().__init__('simulation_tester')
        
        self.received = {
            'joint_states': False,
            'camera_left': False,
            'camera_bottom': False,
            'camera_right': False,
            'ft_sensor': False,
        }
        
        # Subscribers
        self.create_subscription(JointState, '/joint_states', self.joint_states_cb, 10)
        self.create_subscription(Image, '/camera_left/image_raw', self.camera_left_cb, 10)
        self.create_subscription(Image, '/camera_bottom/image_raw', self.camera_bottom_cb, 10)
        self.create_subscription(Image, '/camera_right/image_raw', self.camera_right_cb, 10)
        self.create_subscription(WrenchStamped, '/ft_sensor', self.ft_sensor_cb, 10)
        
        # Timer to check status
        self.create_timer(2.0, self.check_status)
        self.get_logger().info('Simulation tester started. Waiting for topics...')
    
    def joint_states_cb(self, msg):
        if not self.received['joint_states']:
            self.get_logger().info(f'✓ Joint states received: {len(msg.name)} joints')
            self.received['joint_states'] = True
    
    def camera_left_cb(self, msg):
        if not self.received['camera_left']:
            self.get_logger().info(f'✓ Camera LEFT: {msg.width}x{msg.height}')
            self.received['camera_left'] = True
    
    def camera_bottom_cb(self, msg):
        if not self.received['camera_bottom']:
            self.get_logger().info(f'✓ Camera BOTTOM: {msg.width}x{msg.height}')
            self.received['camera_bottom'] = True
    
    def camera_right_cb(self, msg):
        if not self.received['camera_right']:
            self.get_logger().info(f'✓ Camera RIGHT: {msg.width}x{msg.height}')
            self.received['camera_right'] = True
    
    def ft_sensor_cb(self, msg):
        if not self.received['ft_sensor']:
            force = msg.wrench.force
            self.get_logger().info(f'✓ F/T Sensor: force=({force.x:.2f}, {force.y:.2f}, {force.z:.2f})')
            self.received['ft_sensor'] = True
    
    def check_status(self):
        all_received = all(self.received.values())
        missing = [k for k, v in self.received.items() if not v]
        
        if all_received:
            self.get_logger().info('\n' + '='*50)
            self.get_logger().info('ALL TOPICS VERIFIED! Simulation is working.')
            self.get_logger().info('='*50)
            rclpy.shutdown()
        else:
            self.get_logger().warn(f'Still waiting for: {missing}')


def main():
    rclpy.init()
    tester = SimulationTester()
    
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
