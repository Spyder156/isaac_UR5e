#!/usr/bin/env python
"""Debug script to print UR5e robot structure - body names, joint names, etc."""

import sys
import argparse

# Parse args before imports
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

# IsaacSim imports
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import torch
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.sim import SimulationContext, SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import spawn_ground_plane

# Import our robot config
from rwm_ur5e.configs.ur5e_reach_cfg import UR5E_CFG

def main():
    # Setup simulation
    sim = SimulationContext(SimulationCfg(dt=1/60, device="cuda:0"))
    
    # Create scene
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    
    # Spawn ground
    spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
    
    # Create robot with explicit prim path
    robot_cfg = UR5E_CFG.replace(prim_path="/World/envs/env_0/Robot")
    robot = Articulation(robot_cfg)
    scene.articulations["robot"] = robot
    
    # Clone and filter (even for 1 env)
    scene.clone_environments(copy_from_source=False)
    scene.filter_collisions(global_prim_paths=[])
    
    # Reset sim
    sim.reset()
    scene.reset()
    
    print("\n" + "="*60)
    print("UR5e ROBOT STRUCTURE DEBUG")
    print("="*60)
    
    # Print body names
    print("\n--- BODY NAMES ---")
    body_names = robot.body_names
    for i, name in enumerate(body_names):
        print(f"  [{i}]: {name}")
    
    # Print joint names
    print("\n--- JOINT NAMES ---")
    joint_names = robot.joint_names
    for i, name in enumerate(joint_names):
        print(f"  [{i}]: {name}")
    
    # Print current joint positions
    print("\n--- CURRENT JOINT POSITIONS (radians) ---")
    joint_pos = robot.data.joint_pos[0]
    for i, (name, pos) in enumerate(zip(joint_names[:6], joint_pos[:6])):
        print(f"  {name}: {pos.item():.4f} rad ({pos.item() * 180/3.14159:.1f}°)")
    
    # Print joint limits
    print("\n--- JOINT LIMITS (radians) ---")
    soft_limits = robot.data.soft_joint_pos_limits[0]
    for i, name in enumerate(joint_names[:6]):
        lo = soft_limits[i, 0].item()
        hi = soft_limits[i, 1].item()
        print(f"  {name}: [{lo:.3f}, {hi:.3f}] rad")
    
    # Print body positions
    print("\n--- BODY POSITIONS (world frame) ---")
    body_pos = robot.data.body_pos_w[0]
    for i, name in enumerate(body_names):
        pos = body_pos[i]
        print(f"  {name}: ({pos[0].item():.4f}, {pos[1].item():.4f}, {pos[2].item():.4f})")
    
    # Find likely EE candidates
    print("\n--- LIKELY END-EFFECTOR CANDIDATES ---")
    for i, name in enumerate(body_names):
        if "link" in name.lower() and ("ee" in name.lower() or "wrist" in name.lower() or "tool" in name.lower()):
            pos = body_pos[i]
            print(f"  {name}: ({pos[0].item():.4f}, {pos[1].item():.4f}, {pos[2].item():.4f})")
    
    # Clean up
    sim.stop()
    simulation_app.close()
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
