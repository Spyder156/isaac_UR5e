# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP observation terms for UR5e RWM."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Joint positions in radians. Shape: (num_envs, 6)."""
    return env.scene["robot"].data.joint_pos[:, :6]


def joint_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Joint velocities in rad/s. Shape: (num_envs, 6)."""
    return env.scene["robot"].data.joint_vel[:, :6]


def ee_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End-effector position in world frame. Shape: (num_envs, 3)."""
    # Get wrist_3_link body position (end-effector)
    ee_idx = env.scene["robot"].find_bodies("wrist_3_link")[0][0]
    return env.scene["robot"].data.body_pos_w[:, ee_idx, :]


def cube_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cube target position in world frame. Shape: (num_envs, 3).
    
    Returns position relative to environment origin for cleaner observations.
    """
    cube_pos_w = env.scene["cube"].data.root_pos_w
    return cube_pos_w - env.scene.env_origins


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous action taken. Shape: (num_envs, 6)."""
    return env.action_manager.action[:, :6]


# Additional utility functions for RWM

def base_lin_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Base linear velocity (for mobile robots, zeros for UR5e). Shape: (num_envs, 3)."""
    return torch.zeros(env.num_envs, 3, device=env.device)


def base_ang_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Base angular velocity (for mobile robots, zeros for UR5e). Shape: (num_envs, 3)."""
    return torch.zeros(env.num_envs, 3, device=env.device)


def projected_gravity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Projected gravity vector (constant for fixed-base robot). Shape: (num_envs, 3)."""
    return torch.tensor([[0.0, 0.0, -1.0]], device=env.device).expand(env.num_envs, -1)
