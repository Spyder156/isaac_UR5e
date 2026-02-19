"""UR5e Cable Plug Environment: Grasp connector + insert into socket.

Multi-phase curriculum:
  Phase 1 (reach+grasp): approach connector, close gripper, lift
  Phase 2 (transport): carry grasped connector toward socket
  Phase 3 (insertion): align and insert connector into socket slot

Uses wrist camera images + proprioception (no ground truth target position).
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

from rwm_ur5e.configs.ur5e_cable_plug_cfg import UR5eCablePlugEnvCfg


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


class UR5eCablePlugEnv(DirectRLEnv):
    cfg: UR5eCablePlugEnvCfg

    def __init__(self, cfg: UR5eCablePlugEnvCfg, render_mode: str | None = None, **kwargs):
        # Generate URDFs before super().__init__ (which calls _setup_scene)
        urdf_dir = os.path.join(PROJECT_ROOT, "rwm_ur5e", "urdf")
        robot_urdf = _generate_urdf(
            os.path.join(urdf_dir, "ur5e_hande_isaac.urdf.xacro"),
            "ur5e_hande_isaac.urdf",
        )
        cable_urdf = _generate_urdf(
            os.path.join(urdf_dir, "cable_rigid_isaac.urdf.xacro"),
            "cable_rigid_isaac.urdf",
        )
        cfg.robot_cfg.spawn.asset_path = robot_urdf
        cfg.cable_cfg.spawn.asset_path = cable_urdf

        super().__init__(cfg, render_mode, **kwargs)

        self._ee_body_idx = self.robot.find_bodies(self.cfg.ee_link_name)[0][0]
        self._wrist_body_idx = self.robot.find_bodies(self.cfg.wrist_link_name)[0][0]
        self._connector_body_idx = self.cable.find_bodies(self.cfg.connector_body_name)[0][0]
        self._grip_body_idx = self.cable.find_bodies(self.cfg.connector_grip_body_name)[0][0]

        # Joint limits for UR5e (first 6 joints only)
        self._joint_pos_min = self.robot.data.soft_joint_pos_limits[:, :6, 0]
        self._joint_pos_max = self.robot.data.soft_joint_pos_limits[:, :6, 1]

        # Action buffers (7D: 6 arm + 1 gripper)
        self._last_action = torch.zeros(self.num_envs, 7, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, 7, device=self.device)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)

        # Grasp state tracking
        self._is_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Initial cable connector Z for lift detection
        self._initial_grip_z = torch.full(
            (self.num_envs,), self.cfg.cable_cfg.init_state.pos[2], device=self.device
        )

        # Socket slot center (world frame offset added per-env in _reset_idx)
        self._socket_center = torch.tensor(
            self.cfg.socket_pos, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1).clone()

        # Curriculum phase tracking
        self._phase = 1
        self._total_steps = 0

        print(f"[INFO] UR5eCablePlugEnv initialized: {self.num_envs} envs")
        print(f"[INFO] EE body: {self._ee_body_idx}, Grip body: {self._grip_body_idx}, "
              f"Connector body: {self._connector_body_idx}")
        print(f"[INFO] Socket center: {self.cfg.socket_pos}")
        print(f"[INFO] Curriculum: phase1 until iter {self.cfg.phase_1_end}, "
              f"phase2 until iter {self.cfg.phase_2_end}, then phase3")

    def _update_curriculum(self):
        """Auto-update curriculum phase based on total env steps."""
        self._total_steps += 1
        # Estimate iteration: total_steps / num_steps_per_env (each call = 1 step)
        # Runner does num_steps_per_env calls per iteration
        # Use 32 as default (matches UR5eCablePlugRWMRunnerCfg.num_steps_per_env)
        estimated_iter = self._total_steps // 32
        if estimated_iter < self.cfg.phase_1_end:
            new_phase = 1
        elif estimated_iter < self.cfg.phase_2_end:
            new_phase = 2
        else:
            new_phase = 3
        if new_phase != self._phase:
            print(f"[CURRICULUM] Phase {self._phase} -> {new_phase} "
                  f"(step {self._total_steps}, ~iter {estimated_iter})", flush=True)
            self._phase = new_phase

    def _setup_scene(self):
        # Ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=sim_utils.GroundPlaneCfg())

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Table
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

        # Circuit board (PCB)
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

        # --- Socket: U-channel composed of multiple cuboids ---
        sx, sy, sz = self.cfg.socket_pos
        wall_t = 0.0025  # wall thickness
        slot_w = self.cfg.socket_slot_width  # 13mm interior
        slot_h = self.cfg.socket_slot_height  # 5mm interior
        slot_d = self.cfg.socket_slot_depth   # 24mm depth

        socket_color = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9))

        # Floor of the slot
        cfg_floor = sim_utils.CuboidCfg(
            size=(slot_w + 2 * wall_t, slot_d, 0.002),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=socket_color,
        )
        cfg_floor.func(
            "/World/envs/env_.*/SocketFloor", cfg_floor,
            translation=(sx, sy, sz - 0.001),
        )

        # Left wall
        cfg_lwall = sim_utils.CuboidCfg(
            size=(wall_t, slot_d, slot_h),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=socket_color,
        )
        cfg_lwall.func(
            "/World/envs/env_.*/SocketLWall", cfg_lwall,
            translation=(sx - (slot_w + wall_t) / 2, sy, sz + slot_h / 2 - 0.001),
        )

        # Right wall
        cfg_rwall = sim_utils.CuboidCfg(
            size=(wall_t, slot_d, slot_h),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=socket_color,
        )
        cfg_rwall.func(
            "/World/envs/env_.*/SocketRWall", cfg_rwall,
            translation=(sx + (slot_w + wall_t) / 2, sy, sz + slot_h / 2 - 0.001),
        )

        # Back wall
        cfg_bwall = sim_utils.CuboidCfg(
            size=(slot_w, wall_t, slot_h),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=socket_color,
        )
        cfg_bwall.func(
            "/World/envs/env_.*/SocketBWall", cfg_bwall,
            translation=(sx, sy - (slot_d + wall_t) / 2, sz + slot_h / 2 - 0.001),
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

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # Register sensors AFTER cloning
        self.scene.sensors["cam_top"] = self._cam_top
        self.scene.sensors["cam_left"] = self._cam_left
        self.scene.sensors["cam_right"] = self._cam_right

    def _pre_physics_step(self, actions: torch.Tensor):
        self._update_curriculum()
        self._prev_action = self._last_action.clone()
        self._last_action = actions.clone()

        # Arm: first 6 dims (delta position control)
        delta = actions[:, :6] * self.cfg.action_scale
        current_pos = self.robot.data.joint_pos[:, :6]
        target_pos = current_pos + delta
        target_pos = torch.clamp(target_pos, self._joint_pos_min, self._joint_pos_max)
        self.robot.set_joint_position_target(target_pos, joint_ids=list(range(6)))

        # Gripper: 7th dim, binary open/close
        # action > 0 → open (0.025m), action ≤ 0 → close (0.0m)
        gripper_cmd = actions[:, 6]  # (N,)
        gripper_open = (gripper_cmd > 0).float()
        gripper_pos = gripper_open.unsqueeze(-1) * self.cfg.gripper_open_pos  # (N, 1)
        gripper_both = gripper_pos.expand(-1, 2)  # same target for both fingers
        self.robot.set_joint_position_target(gripper_both, joint_ids=[6, 7])

    def _apply_action(self):
        pass

    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos = self.robot.data.joint_pos[:, :6]
        joint_vel = self.robot.data.joint_vel[:, :6]

        ee_pos_world = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        ee_pos = ee_pos_world - self.scene.env_origins

        # Gripper state: average of both finger positions, normalized to [0, 1]
        gripper_left = self.robot.data.joint_pos[:, 6:7]
        gripper_right = self.robot.data.joint_pos[:, 7:8]
        gripper_state = ((gripper_left + gripper_right) / 2.0) / self.cfg.gripper_open_pos  # (N, 1)

        # Camera images
        img_top = self._cam_top.data.output["rgb"][:, :, :, :3].float() / 255.0
        img_left = self._cam_left.data.output["rgb"][:, :, :, :3].float() / 255.0
        img_right = self._cam_right.data.output["rgb"][:, :, :, :3].float() / 255.0

        images = torch.cat([
            img_top.reshape(self.num_envs, -1),
            img_left.reshape(self.num_envs, -1),
            img_right.reshape(self.num_envs, -1),
        ], dim=-1)

        # Camera world poses
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
                gripper_state,       # 1
                self._last_action,   # 7
            ], dim=-1),              # total: 23
            "policy_image": images,
            "camera_poses": camera_poses,
            "system_state": torch.cat([
                joint_pos,           # 6
                joint_vel,           # 6
                ee_pos,              # 3
            ], dim=-1),              # total: 15
            "system_action": self._last_action,  # 7
        }

    def _update_grasp_state(self):
        """Detect if connector is grasped using physics-based heuristics."""
        # Gripper is commanded closed
        gripper_closed = self._last_action[:, 6] <= 0

        # Connector grip is near EE
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        grip_pos = self.cable.data.body_pos_w[:, self._grip_body_idx, :]
        grip_dist = torch.norm(grip_pos - ee_pos, dim=-1)
        grip_near = grip_dist < self.cfg.grasp_dist_threshold

        # Connector is lifted above initial position
        grip_z = grip_pos[:, 2]
        grip_lifted = grip_z > (self._initial_grip_z + self.cfg.grasp_lift_threshold)

        self._is_grasped = gripper_closed & grip_near & grip_lifted

    def _get_rewards(self) -> torch.Tensor:
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        wrist_pos = self.robot.data.body_pos_w[:, self._wrist_body_idx, :]
        grip_pos = self.cable.data.body_pos_w[:, self._grip_body_idx, :]
        connector_pos = self.cable.data.body_pos_w[:, self._connector_body_idx, :]

        # Update grasp detection
        self._update_grasp_state()

        # Socket center in world frame
        socket_center_w = self._socket_center + self.scene.env_origins

        # --- Phase 1: Reach + Grasp ---
        dist_to_grip = torch.norm(ee_pos - grip_pos, dim=-1)
        reach_reward = 1.0 / (1.0 + 10.0 * dist_to_grip)

        # Grasp bonus
        grasp_reward = self._is_grasped.float() * 5.0

        # --- Phase 2: Transport ---
        dist_to_socket = torch.norm(connector_pos - socket_center_w, dim=-1)
        transport_reward = self._is_grasped.float() * 1.0 / (1.0 + 10.0 * dist_to_socket)

        # --- Phase 3: Insertion ---
        near_socket = dist_to_socket < 0.05  # within 5cm
        insertion_reward = (self._is_grasped & near_socket).float() * 1.0 / (1.0 + 50.0 * dist_to_socket)

        # --- Posture: gripper pointing down ---
        wrist_to_ee = ee_pos - wrist_pos
        wrist_to_ee_len = torch.norm(wrist_to_ee, dim=-1, keepdim=True).clamp(min=1e-6)
        wrist_to_ee_dir = wrist_to_ee / wrist_to_ee_len
        downward_dot = -wrist_to_ee_dir[:, 2]
        posture_reward = 0.3 * torch.clamp(downward_dot, min=0.0)

        # --- Action penalty ---
        action_magnitude = torch.norm(self._last_action[:, :6], dim=-1)
        action_penalty = 0.05 * action_magnitude

        # --- Combine based on curriculum phase ---
        if self._phase == 1:
            reward = reach_reward + grasp_reward + posture_reward - action_penalty
        elif self._phase == 2:
            reward = reach_reward + grasp_reward + transport_reward + posture_reward - action_penalty
        else:  # phase 3
            reward = (
                reach_reward + grasp_reward + transport_reward
                + insertion_reward + posture_reward - action_penalty
            )
            # Insertion success bonus
            inserted = dist_to_socket < self.cfg.success_threshold
            reward = torch.where(
                inserted & self._is_grasped,
                reward + self.cfg.success_bonus,
                reward,
            )

        self._prev_dist = dist_to_grip.clone()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Truncated: timeout
        truncated = self.episode_length_buf >= self.max_episode_length

        # Terminated: insertion success (phase 3 only)
        if self._phase >= 3:
            connector_pos = self.cable.data.body_pos_w[:, self._connector_body_idx, :]
            socket_center_w = self._socket_center + self.scene.env_origins
            dist_to_socket = torch.norm(connector_pos - socket_center_w, dim=-1)
            terminated = (dist_to_socket < self.cfg.success_threshold) & self._is_grasped
        else:
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        # Reset robot to home pose (includes gripper open)
        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        default_joint_vel = torch.zeros_like(default_joint_pos)
        self.robot.write_joint_state_to_sim(
            default_joint_pos, default_joint_vel, env_ids=env_ids,
        )

        # Reset cable to randomized position
        num_resets = len(env_ids)
        cable_default_state = self.cable.data.default_root_state[env_ids].clone()
        # Randomize XY within a region on the table (default pos is 0.2, -0.1)
        # X: [0.10, 0.35], Y: [-0.20, 0.05] — reachable workspace, away from socket
        cable_rand_x = torch.rand(num_resets, device=self.device) * 0.25 + 0.10
        cable_rand_y = torch.rand(num_resets, device=self.device) * 0.25 - 0.20
        cable_default_state[:, 0] = cable_rand_x + self.scene.env_origins[env_ids, 0]
        cable_default_state[:, 1] = cable_rand_y + self.scene.env_origins[env_ids, 1]
        cable_default_state[:, 2] = self.cfg.cable_cfg.init_state.pos[2] + self.scene.env_origins[env_ids, 2]
        # Zero out velocities
        cable_default_state[:, 7:] = 0.0
        self.cable.write_root_state_to_sim(cable_default_state, env_ids=env_ids)

        # Reset cable joints (straight cable)
        cable_default_joint_pos = self.cable.data.default_joint_pos[env_ids]
        cable_default_joint_vel = torch.zeros_like(cable_default_joint_pos)
        self.cable.write_joint_state_to_sim(
            cable_default_joint_pos, cable_default_joint_vel, env_ids=env_ids,
        )

        # Reset action buffers
        self._last_action[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._prev_dist[env_ids] = 0.0
        self._is_grasped[env_ids] = False

        # Store initial grip Z for lift detection
        self._initial_grip_z[env_ids] = self.cfg.cable_cfg.init_state.pos[2]

        # Update socket center for these envs
        self._socket_center[env_ids] = torch.tensor(
            self.cfg.socket_pos, device=self.device
        )
