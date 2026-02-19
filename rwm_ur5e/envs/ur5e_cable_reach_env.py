"""UR5e Cable Reach Environment with Vision Observations.

The robot must reach the USB connector at the end of a rigid cable
using wrist camera images + proprioception. No ground truth target
position is provided to the policy — it must learn to locate the
connector from raw RGB images.

Rewards use GT distance internally (standard practice).
"""

from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import spawn_ground_plane
from isaaclab.utils.math import quat_apply

from rwm_ur5e.configs.ur5e_cable_reach_cfg import UR5eCableReachEnvCfg


# ---------- URDF Helpers (from view_robot.py) ----------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ros_package_path() -> str:
    paths = [
        os.path.join(PROJECT_ROOT, "ur_description"),
        os.path.join(PROJECT_ROOT, "robotiq_hande_description"),
        os.path.join(PROJECT_ROOT, "gazebo_cable_env"),
    ]
    existing = os.environ.get("ROS_PACKAGE_PATH", "")
    if existing:
        paths.append(existing)
    return ":".join(paths)


def _fix_package_paths(urdf_path: str):
    with open(urdf_path, "r") as f:
        content = f.read()
    replacements = {
        "package://robotiq_hande_description/": os.path.join(PROJECT_ROOT, "robotiq_hande_description") + "/",
        "package://ur_description/": os.path.join(PROJECT_ROOT, "ur_description") + "/",
        "package://gazebo_cable_env/": os.path.join(PROJECT_ROOT, "gazebo_cable_env") + "/",
    }
    for old, new in replacements.items():
        content = content.replace(old, new)
    with open(urdf_path, "w") as f:
        f.write(content)


def _generate_urdf(xacro_path: str, output_name: str) -> str:
    urdf_out = os.path.join("/tmp", output_name)
    ros_setup = "/opt/ros/jazzy/setup.bash"
    colcon_setup = os.path.join(PROJECT_ROOT, "install", "setup.bash")
    pkg_path = _ros_package_path()

    cmd = f'source {ros_setup} 2>/dev/null; '
    if os.path.exists(colcon_setup):
        cmd += f'source {colcon_setup} 2>/dev/null; '
    cmd += f'export ROS_PACKAGE_PATH="{pkg_path}:${{ROS_PACKAGE_PATH:-}}"; '
    cmd += f'xacro {xacro_path}'

    result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"xacro failed for {xacro_path}:\n{result.stderr}")

    with open(urdf_out, "w") as f:
        f.write(result.stdout)
    _fix_package_paths(urdf_out)
    print(f"[INFO] Generated URDF: {urdf_out}")
    return urdf_out


# ---------- Environment ----------


class UR5eCableReachEnv(DirectRLEnv):
    cfg: UR5eCableReachEnvCfg

    def __init__(self, cfg: UR5eCableReachEnvCfg, render_mode: str | None = None, **kwargs):
        # Generate URDFs and set paths on config before super().__init__ calls _setup_scene
        robot_xacro = os.path.join(PROJECT_ROOT, "rwm_ur5e", "urdf", "ur5e_hande_isaac.urdf.xacro")
        cable_xacro = os.path.join(PROJECT_ROOT, "rwm_ur5e", "urdf", "cable_rigid_isaac.urdf.xacro")

        robot_urdf = _generate_urdf(robot_xacro, "ur5e_hande_isaac.urdf")
        cable_urdf = _generate_urdf(cable_xacro, "cable_rigid_isaac.urdf")

        cfg.robot_cfg.spawn.asset_path = robot_urdf
        cfg.cable_cfg.spawn.asset_path = cable_urdf

        super().__init__(cfg, render_mode, **kwargs)

        self._ee_body_idx = self.robot.find_bodies(self.cfg.ee_link_name)[0][0]
        self._wrist_body_idx = self.robot.find_bodies(self.cfg.wrist_link_name)[0][0]
        self._connector_body_idx = self.cable.find_bodies(self.cfg.connector_body_name)[0][0]

        # Joint limits for UR5e (first 6 joints only)
        self._joint_pos_min = self.robot.data.soft_joint_pos_limits[:, :6, 0]
        self._joint_pos_max = self.robot.data.soft_joint_pos_limits[:, :6, 1]

        self._last_action = torch.zeros(self.num_envs, 6, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, 6, device=self.device)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)

        print(f"[INFO] UR5eCableReachEnv initialized: {self.num_envs} envs")
        print(f"[INFO] EE body idx: {self._ee_body_idx}, Connector body idx: {self._connector_body_idx}")

    def _setup_scene(self):
        # Ground plane (global — shared across all envs, collides with everything)
        spawn_ground_plane(prim_path="/World/ground", cfg=sim_utils.GroundPlaneCfg())

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Table — per-env with physics collision
        cfg_table = sim_utils.CuboidCfg(
            size=(1.5, 1.0, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.55, 0.45)),
        )
        cfg_table.func(
            "/World/envs/env_.*/Table", cfg_table,
            translation=(0.0, 0.0, self.cfg.table_height),
        )

        # Robot
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # Cable
        self.cable = Articulation(self.cfg.cable_cfg)
        self.scene.articulations["cable"] = self.cable

        # Circuit board
        cfg_pcb = sim_utils.CuboidCfg(
            size=(0.10, 0.08, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
        )
        cfg_pcb.func(
            "/World/envs/env_.*/PCB", cfg_pcb,
            translation=(0.5, -0.25, self.cfg.table_surface_z + 0.005),
        )

        cfg_socket = sim_utils.CuboidCfg(
            size=(0.018, 0.010, 0.016),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
        )
        cfg_socket.func(
            "/World/envs/env_.*/Socket", cfg_socket,
            translation=(0.5, -0.25, self.cfg.table_surface_z + 0.018),
        )

        # Camera housings and support arms (visual only)
        cam_housing_cfg = sim_utils.CuboidCfg(
            size=(0.03, 0.02, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
        )
        arm_thickness = 0.008

        cam_positions = [
            ("cam_top", (0.0, 0.10, 0.03)),
            ("cam_left", (0.10, 0.0, 0.03)),
            ("cam_right", (-0.10, 0.0, 0.03)),
        ]
        for cam_name, (rx, ry, rz) in cam_positions:
            cam_housing_cfg.func(
                f"/World/envs/env_.*/Robot/wrist_3_link/{cam_name}_body", cam_housing_cfg,
                translation=(rx, ry, rz - 0.02),
            )
            arm_len = math.sqrt(rx * rx + ry * ry)
            if abs(rx) > abs(ry):
                arm_size = (arm_len, arm_thickness, arm_thickness)
            else:
                arm_size = (arm_thickness, arm_len, arm_thickness)
            arm_cfg = sim_utils.CuboidCfg(
                size=arm_size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
            )
            arm_cfg.func(
                f"/World/envs/env_.*/Robot/wrist_3_link/{cam_name}_arm", arm_cfg,
                translation=(rx / 2.0, ry / 2.0, rz - 0.02),
            )

        # 3 TiledCameras
        self._cam_top = TiledCamera(self.cfg.cam_top_cfg)
        self._cam_left = TiledCamera(self.cfg.cam_left_cfg)
        self._cam_right = TiledCamera(self.cfg.cam_right_cfg)

        # Clone environments — ground plane is global (collides with all envs)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # Register sensors AFTER cloning
        self.scene.sensors["cam_top"] = self._cam_top
        self.scene.sensors["cam_left"] = self._cam_left
        self.scene.sensors["cam_right"] = self._cam_right

    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_action = self._last_action.clone()
        self._last_action = actions.clone()

        delta = actions * self.cfg.action_scale
        current_pos = self.robot.data.joint_pos[:, :6]
        target_pos = current_pos + delta
        target_pos = torch.clamp(target_pos, self._joint_pos_min, self._joint_pos_max)

        self.robot.set_joint_position_target(target_pos, joint_ids=list(range(6)))

    def _apply_action(self):
        pass

    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos = self.robot.data.joint_pos[:, :6]
        joint_vel = self.robot.data.joint_vel[:, :6]

        ee_pos_world = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        ee_pos = ee_pos_world - self.scene.env_origins

        # Camera images: (N, H, W, 4) uint8 -> (N, H, W, 3) float [0,1]
        img_top = self._cam_top.data.output["rgb"][:, :, :, :3].float() / 255.0
        img_left = self._cam_left.data.output["rgb"][:, :, :, :3].float() / 255.0
        img_right = self._cam_right.data.output["rgb"][:, :, :, :3].float() / 255.0

        # Flatten and concatenate all camera images
        images = torch.cat([
            img_top.reshape(self.num_envs, -1),
            img_left.reshape(self.num_envs, -1),
            img_right.reshape(self.num_envs, -1),
        ], dim=-1)

        # Camera world poses: position (local frame) + forward vector per camera
        # Forward vector = rotate [0,0,1] by camera quaternion (ROS convention: Z forward)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1)

        cam_top_pos = self._cam_top.data.pos_w - self.scene.env_origins
        cam_top_fwd = quat_apply(self._cam_top.data.quat_w_ros, z_axis)
        cam_left_pos = self._cam_left.data.pos_w - self.scene.env_origins
        cam_left_fwd = quat_apply(self._cam_left.data.quat_w_ros, z_axis)
        cam_right_pos = self._cam_right.data.pos_w - self.scene.env_origins
        cam_right_fwd = quat_apply(self._cam_right.data.quat_w_ros, z_axis)

        camera_poses = torch.cat([
            cam_top_pos, cam_top_fwd,
            cam_left_pos, cam_left_fwd,
            cam_right_pos, cam_right_fwd,
        ], dim=-1)  # (N, 18)

        return {
            "policy": torch.cat([
                joint_pos,           # 6
                joint_vel,           # 6
                ee_pos,              # 3
                self._last_action,   # 6
            ], dim=-1),              # total: 21
            "policy_image": images,  # (N, 3*64*64*3) = (N, 36864)
            "camera_poses": camera_poses,  # (N, 18)
            "system_state": torch.cat([
                joint_pos,           # 6
                joint_vel,           # 6
                ee_pos,              # 3
            ], dim=-1),              # total: 15
            "system_action": self._last_action,  # 6
        }

    def _get_rewards(self) -> torch.Tensor:
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        wrist_pos = self.robot.data.body_pos_w[:, self._wrist_body_idx, :]
        connector_pos = self.cable.data.body_pos_w[:, self._connector_body_idx, :]
        dist = torch.norm(ee_pos - connector_pos, dim=-1)

        # --- 1. Distance reward ---
        dist_reward = 1.0 / (1.0 + 10.0 * dist)

        # --- 2. Progress reward ---
        progress = self._prev_dist - dist
        has_valid_prev = self._prev_dist > 0.01
        progress_reward = torch.where(
            has_valid_prev,
            torch.clamp(progress * 10.0, -0.5, 0.5),
            torch.zeros_like(progress),
        )

        # --- 3. Action penalty ---
        action_magnitude = torch.norm(self._last_action, dim=-1)
        closeness = torch.exp(-10.0 * dist)
        action_penalty = action_magnitude * (0.05 + 0.2 * closeness)

        # # --- Jerk penalty (commented out for now) ---
        # action_delta = self._last_action - self._prev_action
        # jerk_penalty = 0.1 * torch.norm(action_delta, dim=-1)

        # --- 4. Stillness bonus near target ---
        is_close = dist < self.cfg.success_threshold
        is_still = action_magnitude < 0.1
        stillness_bonus = torch.where(
            is_close & is_still,
            torch.full_like(dist, 1.5),
            torch.zeros_like(dist),
        )

        # --- 5. Upright posture reward ---
        # The vector from wrist to gripper tip should point DOWN (world -Z).
        # This is how a workstation robot naturally approaches from above.
        wrist_to_ee = ee_pos - wrist_pos  # (N, 3)
        wrist_to_ee_len = torch.norm(wrist_to_ee, dim=-1, keepdim=True).clamp(min=1e-6)
        wrist_to_ee_dir = wrist_to_ee / wrist_to_ee_len  # unit vector
        # dot with -Z: +1 when pointing straight down, -1 when pointing up
        downward_dot = -wrist_to_ee_dir[:, 2]  # (N,)
        # Reward: 0.3 when perfectly downward, 0 when horizontal or worse
        posture_reward = 0.3 * torch.clamp(downward_dot, min=0.0)

        # --- Combine ---
        reward = (
            dist_reward
            + progress_reward
            - action_penalty
            + stillness_bonus
            + posture_reward
        )

        # Success bonus
        success = dist < self.cfg.success_threshold
        reward = torch.where(success, reward + self.cfg.success_bonus, reward)

        self._prev_dist = dist.clone()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        connector_pos = self.cable.data.body_pos_w[:, self._connector_body_idx, :]
        dist = torch.norm(ee_pos - connector_pos, dim=-1)

        terminated = dist < self.cfg.success_threshold
        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        # Reset robot to home pose
        default_joint_pos = self.robot.data.default_joint_pos[env_ids, :6]
        default_joint_vel = torch.zeros_like(default_joint_pos)
        self.robot.write_joint_state_to_sim(
            default_joint_pos, default_joint_vel,
            joint_ids=list(range(6)), env_ids=env_ids,
        )

        # Reset cable to default state
        cable_default_state = self.cable.data.default_root_state[env_ids].clone()
        cable_default_state[:, :3] += self.scene.env_origins[env_ids]
        self.cable.write_root_state_to_sim(cable_default_state, env_ids=env_ids)

        # Reset cable joint positions to zero (straight)
        cable_default_joint_pos = self.cable.data.default_joint_pos[env_ids]
        cable_default_joint_vel = torch.zeros_like(cable_default_joint_pos)
        self.cable.write_joint_state_to_sim(
            cable_default_joint_pos, cable_default_joint_vel,
            env_ids=env_ids,
        )

        self._last_action[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._prev_dist[env_ids] = 0.0
