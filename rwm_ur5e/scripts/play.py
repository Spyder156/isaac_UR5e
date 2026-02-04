import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate trained UR5e policy")
parser.add_argument("--task", type=str, default="RWM-UR5e-Reach-Play-v0", help="Task ID")
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
parser.add_argument("--num_steps", type=int, default=1000, help="Steps to run")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import rwm_ur5e.envs  # noqa: F401
from rwm_ur5e.configs import UR5eReachEnvCfg_PLAY, UR5eReachRWMRunnerCfg


def main():    
    env_cfg = UR5eReachEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    agent_cfg = UR5eReachRWMRunnerCfg()
    
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    
    # Create runner and load checkpoint
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)
    
    policy = runner.get_inference_policy(device="cuda:0")
    
    # Run policy
    obs, _ = env.reset()
    total_reward = 0.0
    
    print(f"Running policy for {args_cli.num_steps} steps...")
    
    for step in range(args_cli.num_steps):
        with torch.no_grad():
            actions = policy(obs)
        
        obs, rewards, dones, truncated, infos = env.step(actions)
        total_reward += rewards.sum().item()
        
        if step % 100 == 0:
            print(f"Step {step}: mean_reward = {rewards.mean().item():.4f}")

    avg_reward = total_reward / args_cli.num_steps / args_cli.num_envs
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()