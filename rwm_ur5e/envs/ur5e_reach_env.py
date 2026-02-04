from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import spawn_ground_plane

from rwm_ur5e.configs.ur5e_reach_cfg import UR5eReachEnvCfg


class UR5eReachEnv(DirectRLEnv):
    cfg: UR5eReachEnvCfg

    def __init__(self, cfg: UR5eReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._ee_body_idx = self.robot.find_bodies(self.cfg.ee_link_name)[0][0]

        # Joint limits for clamping
        self._joint_pos_min = self.robot.data.soft_joint_pos_limits[:, :6, 0]
        self._joint_pos_max = self.robot.data.soft_joint_pos_limits[:, :6, 1]

        # Initialize last action buffer
        self._last_action = torch.zeros(self.num_envs, 6, device=self.device)
        
        # Distance tracking for rewards
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=sim_utils.GroundPlaneCfg())
        
        # Add robot
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        
        # Add cube
        self.cube = RigidObject(self.cfg.cube_cfg)
        self.scene.rigid_objects["cube"] = self.cube
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Apply filters
        self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor):
        # Store for last_action observation
        self._last_action = actions.clone()
        
        # Scale actions to joint position deltas
        delta = actions * self.cfg.action_scale
        
        # Get current positions and compute targets
        current_pos = self.robot.data.joint_pos[:, :6]
        target_pos = current_pos + delta
        
        # Clamp to joint limits
        target_pos = torch.clamp(target_pos, self._joint_pos_min, self._joint_pos_max)
        
        # Apply position targets
        self.robot.set_joint_position_target(target_pos, joint_ids=list(range(6)))
        
        # Store target for _apply_action
        self._target_pos = target_pos

    def _apply_action(self):
        pass

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Get current state
        joint_pos = self.robot.data.joint_pos[:, :6]
        joint_vel = self.robot.data.joint_vel[:, :6]
        
        # IMPORTANT: Convert to local frame (relative to env origin)
        ee_pos_world = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        ee_pos = ee_pos_world - self.scene.env_origins
        
        # Cube position relative to env origin
        cube_pos = self.cube.data.root_pos_w - self.scene.env_origins
        
        # Relative position (EE to cube) - useful for policy
        ee_to_cube = cube_pos - ee_pos
        
        # Build observations dict
        obs = {
            "policy": torch.cat([
                joint_pos,           # 6
                joint_vel,           # 6
                ee_pos,              # 3 (local frame)
                ee_to_cube,          # 3 (relative direction to target!)
                self._last_action,   # 6
            ], dim=-1),
            "system_state": torch.cat([
                joint_pos,           # 6
                joint_vel,           # 6
                ee_pos,              # 3 (local frame)
                cube_pos,            # 3 (local frame)
            ], dim=-1),
            "system_action": self._last_action,  # 6
        }
        
        return obs

    def _get_rewards(self) -> torch.Tensor:
        # Get EE and cube positions (world frame for distance calc)
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        cube_pos = self.cube.data.root_pos_w
        
        # Distance to target
        dist = torch.norm(ee_pos - cube_pos, dim=-1)
        
        # Distance reward: 1/(1+k*dist) gives smoother gradient than exp
        dist_reward = 1.0 / (1.0 + 10.0 * dist)
        
        # Progress reward: bonus for getting closer
        progress = self._prev_dist - dist
        has_valid_prev = self._prev_dist > 0.01
        progress_reward = torch.where(
            has_valid_prev,
            torch.clamp(progress * 10.0, -0.5, 0.5),
            torch.zeros_like(progress)
        )
        
        # ACTION SMOOTHNESS PENALTY - scales up when close to target
        # This discourages jittering near the goal
        action_magnitude = torch.norm(self._last_action, dim=-1)
        
        # When far from target: small penalty (allow movement)
        # When close to target: larger penalty (discourage jitter)
        closeness = torch.exp(-10.0 * dist)  # 1.0 when dist=0, ~0 when dist>0.3
        action_penalty = action_magnitude * (0.05 + 0.2 * closeness)
        
        # STILLNESS BONUS: reward for small actions when very close
        is_close = dist < 0.05  # within 5cm
        is_still = action_magnitude < 0.1  # small action
        stillness_bonus = torch.where(
            is_close & is_still,
            torch.full_like(dist, 0.3),  # bonus for being still near target
            torch.zeros_like(dist)
        )
        
        # Combine rewards
        reward = dist_reward + progress_reward - action_penalty + stillness_bonus
        
        # Big bonus for success
        success = dist < self.cfg.success_threshold
        reward = torch.where(success, reward + self.cfg.success_bonus, reward)
        
        # Store for next step
        self._prev_dist = dist.clone()
        
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Get current distance
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        cube_pos = self.cube.data.root_pos_w
        dist = torch.norm(ee_pos - cube_pos, dim=-1)
        
        # Terminated: success (reached target)
        terminated = dist < self.cfg.success_threshold
        
        # Truncated: timeout
        truncated = self.episode_length_buf >= self.max_episode_length
        
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        
        num_resets = len(env_ids)
        
        # Reset robot to home pose
        default_joint_pos = self.robot.data.default_joint_pos[env_ids, :6]
        default_joint_vel = torch.zeros_like(default_joint_pos)
        
        self.robot.write_joint_state_to_sim(
            default_joint_pos, 
            default_joint_vel, 
            joint_ids=list(range(6)), 
            env_ids=env_ids
        )
        
        # Randomize cube position within workspace
        min_pos = torch.tensor(self.cfg.workspace_min, device=self.device)
        max_pos = torch.tensor(self.cfg.workspace_max, device=self.device)
        
        cube_pos = torch.rand(num_resets, 3, device=self.device) * (max_pos - min_pos) + min_pos
        cube_pos = cube_pos + self.scene.env_origins[env_ids]
        
        # Set cube position
        cube_state = self.cube.data.default_root_state[env_ids].clone()
        cube_state[:, :3] = cube_pos
        self.cube.write_root_state_to_sim(cube_state, env_ids=env_ids)
        
        # Reset action buffer
        self._last_action[env_ids] = 0.0
        
        # IMPORTANT: Initialize _prev_dist to actual distance (fixes first-step reward bug)
        # We need to compute the actual starting distance after reset
        # Use a sentinel value that will be detected in _get_rewards
        self._prev_dist[env_ids] = 0.0  # Will be detected and skipped in reward calc
