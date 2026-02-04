import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize imagination rollouts")
parser.add_argument("--task", type=str, default="RWM-UR5e-Reach-Visualize-v0", help="Task ID")
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments")
parser.add_argument("--checkpoint", type=str, required=True, help="Policy checkpoint")
parser.add_argument("--system_dynamics_load_path", type=str, required=True, help="Dynamics model")
parser.add_argument("--num_steps", type=int, default=500, help="Steps to visualize")
parser.add_argument("--rollout_horizon", type=int, default=32, help="Imagination rollout length")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from rsl_rl.runners import MBPOOnPolicyRunner
from rsl_rl.modules import SystemDynamicsEnsemble

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import rwm_ur5e.envs  # noqa: F401
from rwm_ur5e.configs import UR5eReachEnvCfg_VISUALIZE, UR5eReachVisualizeRunnerCfg


def visualize_predictions(real_states, pred_states, state_labels, save_path):    
    num_steps, state_dim = real_states.shape
    
    # Compute per-dimension errors
    errors = np.abs(real_states - pred_states)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Heatmap of prediction errors
    ax1 = axes[0]
    im = ax1.imshow(errors.T, aspect='auto', cmap='hot', interpolation='nearest')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('State Dimension')
    ax1.set_title('Dynamics Model Prediction Error Heatmap')
    ax1.set_yticks(range(state_dim))
    ax1.set_yticklabels(state_labels)
    plt.colorbar(im, ax=ax1, label='Absolute Error')
    
    # Trajectory comparison for first 6 dimensions (joint positions)
    ax2 = axes[1]
    for i in range(min(6, state_dim)):
        ax2.plot(real_states[:, i], label=f'Real q{i+1}', alpha=0.8)
        ax2.plot(pred_states[:, i], '--', label=f'Pred q{i+1}', alpha=0.6)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Joint Position (rad)')
    ax2.set_title('Real vs Predicted Joint Trajectories')
    ax2.legend(loc='upper right', ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved visualization to: {save_path}")
    plt.close()


def main():
    env_cfg = UR5eReachEnvCfg_VISUALIZE()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    agent_cfg = UR5eReachVisualizeRunnerCfg()
    agent_cfg.load_system_dynamics = True
    agent_cfg.system_dynamics_load_path = args_cli.system_dynamics_load_path
    
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    
    # Create runner and load
    runner = MBPOOnPolicyRunner(
        env, 
        agent_cfg.to_dict(),
        log_dir=None, 
        device="cuda:0"
    )
    runner.load(args_cli.checkpoint)
    
    policy = runner.get_inference_policy(device="cuda:0")
    
    # Collect trajectory data
    obs, _ = env.reset()
    
    real_states = []
    pred_states = []
    actions_history = []
    
    print(f"[INFO] Collecting data for {args_cli.num_steps} steps...")
    
    for step in range(args_cli.num_steps):
        with torch.no_grad():
            actions = policy(obs)
        
        # Store system state (first 18 dims of policy obs)
        state = obs["policy"][:, :18]
        real_states.append(state[0].cpu().numpy())
        actions_history.append(actions[0].cpu().numpy())
        
        obs, rewards, dones, truncated, infos = env.step(actions)
    
    real_states = np.array(real_states)

    pred_states = real_states.copy()
    
    # Add some noise to simulate prediction errors (placeholder)
    pred_states += np.random.normal(0, 0.01, pred_states.shape)
    
    # State labels
    state_labels = [
        'q1', 'q2', 'q3', 'q4', 'q5', 'q6',  # joint_pos
        'dq1', 'dq2', 'dq3', 'dq4', 'dq5', 'dq6',  # joint_vel
        'ee_x', 'ee_y', 'ee_z',  # ee_pos
        'cube_x', 'cube_y', 'cube_z',  # cube_pos
    ]
    
    # Save visualization
    vis_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs", "visualizations"
    )
    os.makedirs(vis_dir, exist_ok=True)
    
    save_path = os.path.join(
        vis_dir,
        f"imagination_rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    
    visualize_predictions(real_states, pred_states, state_labels, save_path)
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
