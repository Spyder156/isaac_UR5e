"""Scripted oracle for cable insertion data collection.

Bypasses env.step() — directly commands IK → absolute joint targets → sim step.
Records raw joint positions, EE poses, camera frames at every step.
Action labels can be derived from joint sequences at training time.

Uses POSITION-ONLY IK (no orientation constraint) for robustness.
The arm finds a natural configuration to reach each Cartesian waypoint.

Usage:
    python rwm_ur5e/scripts/collect_oracle.py --num_envs 4 --num_episodes 10 --headless --log_phases

State machine (per env, independent):
    ABOVE_GRIP  → hover 15cm above connector grip body
    DESCEND     → lower to 1cm above grip
    GRASP       → close fingers, dwell 50 steps
    LIFT        → raise 15cm
    TRANSPORT   → carry to in-front-and-above socket
    Z_ALIGN     → lower to socket height (still 10cm in front)
    Y_INSERT    → push connector into slot (error-based EE target)
    SUCCESS     → open gripper, episode done
"""

from __future__ import annotations

import argparse
import enum
import os
import sys
import time

RWM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RWM_ROOT)
sys.path.insert(0, os.path.dirname(RWM_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted oracle data collection")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=100, help="Total episodes to collect (across all envs)")
parser.add_argument("--output_dir", type=str, default=os.path.join(RWM_ROOT, "data", "oracle_episodes"))
parser.add_argument("--max_steps_per_episode", type=int, default=600, help="Timeout per episode")
parser.add_argument("--grasp_dwell", type=int, default=20, help="Steps to hold gripper closed before lifting")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_phases", action="store_true", help="Print every phase transition")
parser.add_argument("--warmup_steps", type=int, default=30, help="Sim steps to run before collection (let physics settle)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Lab imports (after sim launch) ──────────────────────────────────────
import numpy as np
import torch

from isaaclab.utils.math import subtract_frame_transforms, quat_mul, quat_conjugate

from rwm_ur5e.envs import register_tasks
register_tasks()

from rwm_ur5e.configs.ur5e_cable_plug_cfg import UR5eCablePlugEnvCfg
from rwm_ur5e.envs.ur5e_cable_plug_env import UR5eCablePlugEnv


# ── Phase enum ─────────────────────────────────────────────────────────────────

class Phase(enum.IntEnum):
    ABOVE_GRIP  = 0
    DESCEND     = 1
    GRASP       = 2
    LIFT        = 3
    TRANSPORT   = 4
    Z_ALIGN     = 5
    Y_INSERT    = 6
    SUCCESS     = 7
    TIMEOUT     = 8


# ── State machine ──────────────────────────────────────────────────────────────

class OracleStateMachine:
    # Positional tolerances (metres)
    THRESH = {
        Phase.ABOVE_GRIP:  0.015,
        Phase.DESCEND:     0.020,   # relaxed: Jacobian-transpose steady-state error ~1-3cm
        Phase.LIFT:        0.015,
        Phase.TRANSPORT:   0.015,
        Phase.Z_ALIGN:     0.020,   # relaxed same reason
    }

    HOVER_Z       = 0.15
    GRASP_OFFSET  = 0.000   # EE targets grip body directly (was 1cm above)
    SOCKET_FRONT  = 0.10   # m in +Y from socket center (approach side)
    SOCKET_ABOVE  = 0.08
    INSERT_THRESH = 0.015

    def __init__(self, num_envs: int, device: str, grasp_dwell: int = 50, verbose: bool = False):
        self.num_envs    = num_envs
        self.device      = device
        self.grasp_dwell = grasp_dwell
        self.verbose     = verbose

        self.phase        = torch.full((num_envs,), Phase.ABOVE_GRIP, dtype=torch.long, device=device)
        self.step_cnt     = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.frozen_target = torch.zeros(num_envs, 3, device=device)

    def reset(self, env_ids: torch.Tensor | list[int]):
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        self.phase[env_ids]         = Phase.ABOVE_GRIP
        self.step_cnt[env_ids]      = 0
        self.frozen_target[env_ids] = 0.0

    def target_pos_world(self, env: UR5eCablePlugEnv) -> torch.Tensor:
        """Return (N, 3) target EE position in world frame for current phase."""
        grip_pos  = env.cable.data.body_pos_w[:, env._grip_body_idx, :]
        ee_pos    = env.robot.data.body_pos_w[:, env._ee_body_idx, :]
        connector = env.cable.data.body_pos_w[:, env._connector_body_idx, :]
        socket_w  = env._socket_center + env.scene.env_origins

        target = torch.zeros(self.num_envs, 3, device=self.device)

        m = self.phase == Phase.ABOVE_GRIP
        if m.any():
            t = grip_pos.clone()
            t[:, 2] += self.HOVER_Z
            target[m] = t[m]

        m = self.phase == Phase.DESCEND
        if m.any():
            # Use frozen target set at ABOVE_GRIP→DESCEND transition; live grip_pos
            # wobbles as the cable is a dynamic articulation and creates a moving target.
            target[m] = self.frozen_target[m]

        m = self.phase == Phase.GRASP
        if m.any():
            target[m] = self.frozen_target[m]

        m = self.phase == Phase.LIFT
        if m.any():
            target[m] = self.frozen_target[m]

        m = self.phase == Phase.TRANSPORT
        if m.any():
            t = socket_w.clone()
            t[:, 1] += self.SOCKET_FRONT
            t[:, 2] += self.SOCKET_ABOVE
            target[m] = t[m]

        m = self.phase == Phase.Z_ALIGN
        if m.any():
            t = socket_w.clone()
            t[:, 1] += self.SOCKET_FRONT
            target[m] = t[m]

        m = self.phase == Phase.Y_INSERT
        if m.any():
            delta = socket_w - connector
            target[m] = (ee_pos + delta)[m]

        m = (self.phase == Phase.SUCCESS) | (self.phase == Phase.TIMEOUT)
        if m.any():
            target[m] = ee_pos[m]

        return target

    def gripper_targets(self) -> torch.Tensor:
        """Return (N, 2) absolute finger joint targets."""
        open_phases = (self.phase == Phase.ABOVE_GRIP) | (self.phase == Phase.DESCEND) \
                    | (self.phase == Phase.SUCCESS) | (self.phase == Phase.TIMEOUT)
        pos = torch.where(open_phases.unsqueeze(-1),
                          torch.full((self.num_envs, 2), 0.025, device=self.device),
                          torch.zeros(self.num_envs, 2, device=self.device))
        return pos

    def update(self, env: UR5eCablePlugEnv, ee_pos_w: torch.Tensor,
               target_pos_w: torch.Tensor, step_num: int):
        dist       = torch.norm(ee_pos_w - target_pos_w, dim=-1)
        grip_pos   = env.cable.data.body_pos_w[:, env._grip_body_idx, :]
        connector  = env.cable.data.body_pos_w[:, env._connector_body_idx, :]
        socket_w   = env._socket_center + env.scene.env_origins
        conn_dist  = torch.norm(connector - socket_w, dim=-1)

        self.step_cnt += 1

        m = (self.phase == Phase.ABOVE_GRIP) & (dist < self.THRESH[Phase.ABOVE_GRIP])
        if m.any():
            # Freeze the descend target NOW so DESCEND phase sees a stable position.
            # live grip_pos wobbles in physics; freezing avoids a moving target.
            t = grip_pos.clone()
            t[:, 2] += self.GRASP_OFFSET
            self.frozen_target[m] = t[m]
            self._transition(m, Phase.DESCEND, f"ABOVE_GRIP→DESCEND (step {step_num})")

        m = (self.phase == Phase.DESCEND) & (dist < self.THRESH[Phase.DESCEND])
        if m.any():
            t = grip_pos.clone()
            t[:, 2] += self.GRASP_OFFSET
            self.frozen_target[m] = t[m]
            self._transition(m, Phase.GRASP, f"DESCEND→GRASP (step {step_num})")

        m = (self.phase == Phase.GRASP) & (self.step_cnt >= self.grasp_dwell)
        if m.any():
            # Check EE is still near grip body — if > 5cm away grasp clearly failed
            grasp_dist = torch.norm(ee_pos_w - grip_pos, dim=-1)
            m_ok   = m & (grasp_dist < 0.05)
            m_fail = m & (grasp_dist >= 0.05)
            if m_ok.any():
                lift_t = self.frozen_target.clone()
                lift_t[:, 2] += self.HOVER_Z
                self.frozen_target[m_ok] = lift_t[m_ok]
                self._transition(m_ok, Phase.LIFT, f"GRASP→LIFT (step {step_num})")
            if m_fail.any():
                self._transition(m_fail, Phase.TIMEOUT,
                                 f"GRASP ABORT: EE drifted {grasp_dist[m_fail].tolist()} m (step {step_num})")

        m = (self.phase == Phase.LIFT) & (dist < self.THRESH[Phase.LIFT])
        if m.any():
            self._transition(m, Phase.TRANSPORT, f"LIFT→TRANSPORT (step {step_num})")

        m = (self.phase == Phase.TRANSPORT) & (dist < self.THRESH[Phase.TRANSPORT])
        if m.any():
            self._transition(m, Phase.Z_ALIGN, f"TRANSPORT→Z_ALIGN (step {step_num})")

        m = (self.phase == Phase.Z_ALIGN) & (dist < self.THRESH[Phase.Z_ALIGN])
        if m.any():
            self._transition(m, Phase.Y_INSERT, f"Z_ALIGN→Y_INSERT (step {step_num})")

        m = (self.phase == Phase.Y_INSERT) & (conn_dist < self.INSERT_THRESH) & env._is_grasped
        if m.any():
            self._transition(m, Phase.SUCCESS, f"Y_INSERT→SUCCESS (step {step_num})")

    def _transition(self, mask: torch.Tensor, new_phase: Phase, msg: str = ""):
        env_ids = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        self.phase[mask]    = new_phase
        self.step_cnt[mask] = 0
        if self.verbose and msg:
            print(f"[Oracle] envs {env_ids}: {msg}", flush=True)


# ── IK helper (Jacobian transpose) ────────────────────────────────────────────

def run_ik(env: UR5eCablePlugEnv, target_pos_w: torch.Tensor,
           home_quat_w: torch.Tensor | None = None,
           gain: float = 3.0, max_step: float = 0.05) -> torch.Tensor:
    """Jacobian-transpose controller (6-DOF when home_quat_w given, 3-DOF otherwise).

    Builds a 6D task-space error [pos_err | 0.3 * orient_err] and multiplies by J^T.
    No matrix inversion — always a descent direction, no singularity blow-up.
    Orientation control is critical: position-only IK lets the wrist drift to random
    angles, and the gripper fingers arrive at the cable with wrong orientation.

    Returns (N, 6) absolute arm joint targets.
    """
    root_pose_w = env.robot.data.root_pose_w                     # (N, 7)
    ee_pose_w   = env.robot.data.body_pose_w[:, env._ee_body_idx, :]

    # EE pose in robot-base frame
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, :3], root_pose_w[:, 3:7],
        ee_pose_w[:, :3],   ee_pose_w[:, 3:7],
    )

    # Target position in base frame (orientation dummy — only pos used here)
    target_pos_b, _ = subtract_frame_transforms(
        root_pose_w[:, :3], root_pose_w[:, 3:7],
        target_pos_w,
        torch.tensor([1., 0., 0., 0.], device=env.device).expand(env.num_envs, -1),
    )

    pos_error = target_pos_b - ee_pos_b                          # (N, 3)

    J_full = env.robot.root_physx_view.get_jacobians()           # (N, L, 6, D)

    if home_quat_w is not None:
        # Transform home (world) quat into robot base frame
        _, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7],
            target_pos_w, home_quat_w,
        )
        # Quaternion error: target ⊗ conj(current) — shortest-path convention
        err_q = quat_mul(target_quat_b, quat_conjugate(ee_quat_b))
        err_q = torch.where(err_q[:, 0:1] < 0, -err_q, err_q)   # ensure w >= 0
        orient_error = 2.0 * err_q[:, 1:]                        # (N, 3) axis-angle approx

        J_ee    = J_full[:, env._ee_body_idx - 1, :, :6]         # (N, 6, 6) full Jacobian
        error_6d = torch.cat([pos_error, 0.3 * orient_error], dim=-1)  # (N, 6)
        dq = torch.bmm(J_ee.transpose(-2, -1),
                       error_6d.unsqueeze(-1)).squeeze(-1)        # (N, 6)
    else:
        J_ee = J_full[:, env._ee_body_idx - 1, :3, :6]           # (N, 3, 6) pos only
        dq   = torch.bmm(J_ee.transpose(-2, -1),
                         pos_error.unsqueeze(-1)).squeeze(-1)     # (N, 6)

    dq = torch.clamp(gain * dq, -max_step, max_step)
    return env.robot.data.joint_pos[:, :6] + dq


# ── Recording ──────────────────────────────────────────────────────────────────

def record_step(env: UR5eCablePlugEnv, oracle: OracleStateMachine) -> dict:
    ee_pose   = env.robot.data.body_pose_w[:, env._ee_body_idx, :]
    img_top   = env._cam_top.data.output["rgb"][:, :, :, :3].cpu().numpy().astype(np.uint8)
    img_left  = env._cam_left.data.output["rgb"][:, :, :, :3].cpu().numpy().astype(np.uint8)
    img_right = env._cam_right.data.output["rgb"][:, :, :, :3].cpu().numpy().astype(np.uint8)
    connector = env.cable.data.body_pos_w[:, env._connector_body_idx, :]
    socket_w  = env._socket_center + env.scene.env_origins
    gripper_l = env.robot.data.joint_pos[:, 6:7]
    gripper_r = env.robot.data.joint_pos[:, 7:8]
    gripper_state = (gripper_l + gripper_r) / 2.0 / env.cfg.gripper_open_pos

    return {
        "joint_pos":     env.robot.data.joint_pos.cpu().numpy(),
        "ee_pose_world": ee_pose.cpu().numpy(),
        "gripper_state": gripper_state.cpu().numpy(),
        "cam_top_rgb":   img_top,
        "cam_left_rgb":  img_left,
        "cam_right_rgb": img_right,
        "connector_pos": connector.cpu().numpy(),
        "socket_pos":    socket_w.cpu().numpy(),
        "oracle_phase":  oracle.phase.cpu().numpy(),
    }


def save_episode(steps: list[dict], env_idx: int, episode_idx: int,
                 success: bool, output_dir: str):
    if not steps:
        return
    stacked = {k: np.stack([s[k][env_idx] for s in steps], axis=0) for k in steps[0]}
    stacked["success"]     = np.array(success)
    stacked["episode_idx"] = np.array(episode_idx)
    label = "success" if success else "fail"
    fname = os.path.join(output_dir, f"ep_{episode_idx:05d}_{label}.npz")
    np.savez_compressed(fname, **stacked)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(args_cli.seed)
    os.makedirs(args_cli.output_dir, exist_ok=True)

    # --- Env ---
    cfg = UR5eCablePlugEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device     = "cuda:0"
    env = UR5eCablePlugEnv(cfg, render_mode=None)
    env.reset()

    print(f"[Oracle] Env ready: {env.num_envs} envs")
    print(f"[Oracle] EE body idx: {env._ee_body_idx}, Grip body: {env._grip_body_idx}")

    # --- Warm-up: let physics settle before starting ---
    print(f"[Oracle] Warming up for {args_cli.warmup_steps} steps...")
    home_joint_pos = env.robot.data.joint_pos[:, :6].clone()
    home_gripper   = torch.full((env.num_envs, 2), 0.025, device=env.device)
    for _ in range(args_cli.warmup_steps):
        env.robot.set_joint_position_target(home_joint_pos, joint_ids=list(range(6)))
        env.robot.set_joint_position_target(home_gripper, joint_ids=[6, 7])
        env.scene.write_data_to_sim()
        env.sim.step()
    env.scene.update(dt=env.cfg.sim.dt * env.cfg.decimation)

    # Debug: diagnostics after warm-up
    ee_pos_now   = env.robot.data.body_pos_w[0, env._ee_body_idx, :]
    grip_pos_now = env.cable.data.body_pos_w[0, env._grip_body_idx, :]
    J_shape = env.robot.root_physx_view.get_jacobians().shape
    ee_quat_now = env.robot.data.body_quat_w[0, env._ee_body_idx, :]
    print(f"[Oracle] Jacobian shape: {J_shape}  (expected: N × links × 6 × dofs)")
    print(f"[Oracle] Using Jacobian row: {env._ee_body_idx - 1} for EE body {env._ee_body_idx}")
    print(f"[Oracle] EE pos  (env 0) after warmup: {ee_pos_now.cpu().tolist()}")
    print(f"[Oracle] EE quat (env 0) [w,x,y,z]:    {ee_quat_now.cpu().tolist()}")
    print(f"[Oracle] Grip pos (env 0):               {grip_pos_now.cpu().tolist()}")
    print(f"[Oracle] ABOVE_GRIP target (env 0):      {(grip_pos_now + torch.tensor([0,0,0.15], device=env.device)).cpu().tolist()}")
    print(f"[Oracle] env_origins (env 0):            {env.scene.env_origins[0].cpu().tolist()}")

    # Record home EE orientation — used to lock wrist during all IK calls.
    # Position-only IK lets wrist drift, so gripper arrives with random finger angle.
    home_quat_w = env.robot.data.body_quat_w[:, env._ee_body_idx, :].clone()  # (N, 4)
    print(f"[Oracle] Home quat locked for orientation control (env 0): {home_quat_w[0].cpu().tolist()}")

    # --- Oracle ---
    oracle = OracleStateMachine(
        num_envs=env.num_envs,
        device=env.device,
        grasp_dwell=args_cli.grasp_dwell,
        verbose=args_cli.log_phases,
    )

    # --- Collection ---
    episode_buffers  = [[] for _ in range(env.num_envs)]
    episode_step_cnt = [0] * env.num_envs
    episodes_done    = [0] * env.num_envs
    total_success    = 0
    total_episodes   = 0
    global_step      = 0

    target_per_env = max(1, args_cli.num_episodes // env.num_envs)
    print(f"[Oracle] Collecting {target_per_env} episodes per env")
    print(f"[Oracle] Output: {args_cli.output_dir}")

    t_start = time.time()

    while min(episodes_done) < target_per_env:

        # ── IK: compute arm joint targets (6D Jacobian transpose) ──
        target_pos_w  = oracle.target_pos_world(env)
        joint_targets = run_ik(env, target_pos_w, home_quat_w=home_quat_w)

        # ── Apply to robot ──
        gripper_t = oracle.gripper_targets()
        env.robot.set_joint_position_target(joint_targets, joint_ids=list(range(6)))
        env.robot.set_joint_position_target(gripper_t, joint_ids=[6, 7])

        # ── Step sim: write targets → physics, then advance ──
        for _ in range(env.cfg.decimation):
            env.scene.write_data_to_sim()
            env.sim.step()
        env.scene.update(dt=env.cfg.sim.dt * env.cfg.decimation)

        # ── Update grasp detection ──
        gripper_closed = gripper_t[:, 0] <= 0.001
        ee_pos_w       = env.robot.data.body_pos_w[:, env._ee_body_idx, :]
        grip_pos_w     = env.cable.data.body_pos_w[:, env._grip_body_idx, :]
        grip_dist      = torch.norm(grip_pos_w - ee_pos_w, dim=-1)
        grip_near      = grip_dist < env.cfg.grasp_dist_threshold
        # Use world-frame initial Z (add env_origin Z to the stored local-frame value)
        grip_initial_z_w = env._initial_grip_z + env.scene.env_origins[:, 2]
        grip_lifted    = grip_pos_w[:, 2] > (grip_initial_z_w + env.cfg.grasp_lift_threshold)
        env._is_grasped = gripper_closed & grip_near & grip_lifted

        # ── Record ──
        step_data = record_step(env, oracle)

        # ── Update oracle phases ──
        oracle.update(env, ee_pos_w, target_pos_w, global_step)
        global_step += 1

        # ── Periodic status print (every 100 steps) ──
        if global_step % 100 == 0:
            phases = [Phase(p.item()).name for p in oracle.phase]
            ee_to_tgt = torch.norm(ee_pos_w - target_pos_w, dim=-1)
            print(f"[Oracle] step={global_step:5d}  phases={phases}  "
                  f"ee→tgt={ee_to_tgt.cpu().tolist()}  "
                  f"grasped={env._is_grasped.cpu().tolist()}  "
                  f"eps_done={episodes_done}", flush=True)

        # ── Per-env episode management ──
        for i in range(env.num_envs):
            if episodes_done[i] >= target_per_env:
                continue

            episode_buffers[i].append(step_data)
            episode_step_cnt[i] += 1

            phase_i   = oracle.phase[i].item()
            timed_out = episode_step_cnt[i] >= args_cli.max_steps_per_episode

            if phase_i == Phase.SUCCESS or timed_out:
                success_i = (phase_i == Phase.SUCCESS)
                if timed_out and phase_i != Phase.SUCCESS:
                    oracle.phase[i] = Phase.TIMEOUT

                ep_idx = total_episodes
                save_episode(episode_buffers[i], env_idx=i,
                             episode_idx=ep_idx, success=success_i,
                             output_dir=args_cli.output_dir)

                label = "SUCCESS" if success_i else f"FAIL@phase{phase_i}"
                print(f"[Oracle] ep={ep_idx:04d} env={i} {label} "
                      f"steps={episode_step_cnt[i]} "
                      f"(done: {episodes_done[i]+1}/{target_per_env})", flush=True)

                if success_i:
                    total_success += 1
                total_episodes += 1

                env_id_tensor = torch.tensor([i], device=env.device)
                env._reset_idx(env_id_tensor)
                oracle.reset([i])
                episode_buffers[i]  = []
                episode_step_cnt[i] = 0
                episodes_done[i]   += 1

    elapsed = time.time() - t_start
    print(f"\n[Oracle] Done. {total_success}/{total_episodes} successful "
          f"({100*total_success/max(1,total_episodes):.1f}%) "
          f"in {elapsed:.1f}s  ({global_step/elapsed:.0f} steps/s)")
    print(f"[Oracle] Episodes saved to: {args_cli.output_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
