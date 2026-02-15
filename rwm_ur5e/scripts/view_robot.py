"""Cable Insertion Scene Viewer - Isaac Sim / IsaacLab

Shows UR5e + Robotiq Hand-E on a table with:
  - Rigid-segment cable (articulated chain)
  - Circuit board with USB insertion port
  - 3 wrist cameras (top, left, right) on the gripper

Usage:
    ./IsaacLab/isaaclab.sh -p rwm_ur5e/scripts/view_robot.py --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cable insertion scene viewer for UR5e + Robotiq Hand-E")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import subprocess

import torch

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import Camera, CameraCfg

# ---------- Constants ----------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLE_HEIGHT = 0.4
TABLE_SURFACE_Z = TABLE_HEIGHT + 0.025  # 0.425m


# ---------- URDF Helpers ----------


def _ros_package_path() -> str:
    """Build ROS_PACKAGE_PATH so xacro can resolve $(find ...) directives."""
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
    """Replace package:// URIs with absolute paths so Isaac Sim URDF converter can find meshes."""
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
    """Run xacro to produce a flat URDF, then fix package:// paths. Returns path to URDF file."""
    urdf_out = os.path.join("/tmp", output_name)

    # Source ROS and colcon workspace, then run xacro.
    # The colcon install/setup.bash provides ament_index so $(find ...) resolves correctly.
    ros_setup = "/opt/ros/jazzy/setup.bash"
    colcon_setup = os.path.join(PROJECT_ROOT, "install", "setup.bash")
    pkg_path = _ros_package_path()

    cmd = f'source {ros_setup} 2>/dev/null; '
    if os.path.exists(colcon_setup):
        cmd += f'source {colcon_setup} 2>/dev/null; '
    cmd += f'export ROS_PACKAGE_PATH="{pkg_path}:${{ROS_PACKAGE_PATH:-}}"; '
    cmd += f'xacro {xacro_path}'

    result = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"xacro failed for {xacro_path}:\n{result.stderr}")

    with open(urdf_out, "w") as f:
        f.write(result.stdout)

    _fix_package_paths(urdf_out)
    print(f"[INFO] Generated URDF: {urdf_out}")
    return urdf_out


# ---------- Scene Design ----------


def design_scene() -> dict:
    """Build the full cable insertion scene. Returns dict of scene entities."""

    # --- Ground plane ---
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # --- Lights ---
    cfg_light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/Light", cfg_light)

    # --- Table ---
    cfg_table = sim_utils.CuboidCfg(
        size=(1.5, 1.0, 0.05),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.55, 0.45)),
    )
    cfg_table.func("/World/Table", cfg_table, translation=(0.0, 0.0, TABLE_HEIGHT))

    # --- UR5e + Robotiq Hand-E ---
    robot_xacro = os.path.join(PROJECT_ROOT, "rwm_ur5e", "urdf", "ur5e_hande_isaac.urdf.xacro")
    robot_urdf = _generate_urdf(robot_xacro, "ur5e_hande_isaac.urdf")

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=robot_urdf,
            fix_base=True,
            merge_fixed_joints=True,
            self_collision=False,
            replace_cylinders_with_capsules=True,
            joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                    stiffness={
                        "shoulder_.*": 1320.0,
                        "elbow_joint": 600.0,
                        "wrist_.*": 216.0,
                        "hande_.*": 500.0,
                    },
                    damping={
                        "shoulder_.*": 72.66,
                        "elbow_joint": 34.64,
                        "wrist_.*": 29.39,
                        "hande_.*": 20.0,
                    },
                ),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_SURFACE_Z),
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -2.0,
                "elbow_joint": 1.5,
                "wrist_1_joint": -1.07,
                "wrist_2_joint": -1.5708,
                "wrist_3_joint": 0.0,
                "hande_left_finger_joint": 0.003,
                "hande_right_finger_joint": 0.003,
            },
        ),
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_.*"],
                stiffness=1320.0,
                damping=72.66,
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                stiffness=600.0,
                damping=34.64,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_.*"],
                stiffness=216.0,
                damping=29.39,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["hande_.*_finger_joint"],
                stiffness=500.0,
                damping=20.0,
            ),
        },
    )
    robot = Articulation(cfg=robot_cfg)

    # --- Circuit Board with USB Port ---
    # Green PCB
    cfg_pcb = sim_utils.CuboidCfg(
        size=(0.10, 0.08, 0.01),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
    )
    cfg_pcb.func("/World/CircuitBoard/PCB", cfg_pcb, translation=(0.5, -0.25, TABLE_SURFACE_Z + 0.005))

    # White USB socket housing (rectangular, matches connector: 11mm x 3mm plug)
    # Housing is slightly larger than the plug opening
    cfg_socket = sim_utils.CuboidCfg(
        size=(0.018, 0.010, 0.016),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
    )
    cfg_socket.func("/World/CircuitBoard/Socket", cfg_socket, translation=(0.5, -0.25, TABLE_SURFACE_Z + 0.018))

    # Dark rectangular USB port slot (visual only — the insertion target)
    # Matches the plug dimensions: 11mm x 3mm
    cfg_slot = sim_utils.CuboidCfg(
        size=(0.012, 0.004, 0.002),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
    )
    cfg_slot.func("/World/CircuitBoard/PortSlot", cfg_slot, translation=(0.5, -0.25, TABLE_SURFACE_Z + 0.027))

    # --- Rigid-Segment Cable (articulated chain) ---
    cable_xacro = os.path.join(PROJECT_ROOT, "rwm_ur5e", "urdf", "cable_rigid_isaac.urdf.xacro")
    cable_urdf = _generate_urdf(cable_xacro, "cable_rigid_isaac.urdf")

    cable_cfg = ArticulationCfg(
        prim_path="/World/RigidCable",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=cable_urdf,
            fix_base=True,
            merge_fixed_joints=False,  # Must keep all segment links for chain to articulate
            self_collision=False,
            replace_cylinders_with_capsules=True,
            joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                drive_type="force",
                target_type="none",  # Passive - no drive
                gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0001,
                ),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Anchor on table to the right, cable lying flat toward robot
            # Rotate 90° around Y so cable's local +Z extends in world -X direction
            pos=(0.4, -0.15, TABLE_SURFACE_Z + 0.01),
            rot=(0.7071, 0.0, 0.7071, 0.0),  # 90° around Y-axis
        ),
        actuators={},
    )
    rigid_cable = Articulation(cfg=cable_cfg)

    # --- 3 Wrist Cameras on Support Arms ---
    # Attached to wrist_3_link (last revolute-joint body).
    # wrist_3_link frame: X+=left, Y+=up, Z+=front (toward fingertips)
    #
    # Cameras mounted near the wrist (low Z), on support arms, looking forward
    # along +Z toward the workspace with slight inward tilt.
    # Positions: Top (+Y=up), Left (+X=left), Right (-X=right)

    cam_spawn = sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 2.0),
    )

    CAM_Z = 0.03       # Near wrist/coupler level (not at fingertips)
    CAM_R = 0.10       # 10cm radial offset (clear of gripper visual mesh)
    TILT = math.radians(30)  # 30 deg inward convergence (see gripper + workspace)
    half = TILT / 2
    c = math.cos(half)
    s = math.sin(half)

    # Visual geometry for camera housings and support arms
    cam_housing_cfg = sim_utils.CuboidCfg(
        size=(0.03, 0.02, 0.02),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
    )
    arm_thickness = 0.008

    # Camera configs: (name, position, tilt_quaternion)
    # Identity rotation in ROS convention looks along parent +Z (forward).
    # Small tilts rotate the view inward toward the gripper center axis.
    cam_configs = [
        # Top: +Y (up), tilt inward toward -Y = rotation around X by +TILT
        ("cam_top",   (0.0, CAM_R, CAM_Z),   (c, s, 0.0, 0.0)),
        # Left: +X (left), tilt inward toward -X = rotation around Y by -TILT
        ("cam_left",  (CAM_R, 0.0, CAM_Z),   (c, 0.0, -s, 0.0)),
        # Right: -X (right), tilt inward toward +X = rotation around Y by +TILT
        ("cam_right", (-CAM_R, 0.0, CAM_Z),  (c, 0.0, s, 0.0)),
    ]

    cameras_dict = {}
    for cam_name, cam_pos, cam_rot in cam_configs:
        rx, ry, rz = cam_pos

        # Visible camera housing BEHIND the camera (offset -Z so it doesn't block the lens)
        cam_housing_cfg.func(
            f"/World/Robot/wrist_3_link/{cam_name}_body", cam_housing_cfg,
            translation=(rx, ry, rz - 0.02),
        )

        # Support arm: thin rod from gripper body center out to camera
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
            f"/World/Robot/wrist_3_link/{cam_name}_arm", arm_cfg,
            translation=(rx / 2.0, ry / 2.0, rz - 0.02),
        )

        # Camera sensor
        cam = Camera(CameraCfg(
            prim_path=f"/World/Robot/wrist_3_link/{cam_name}",
            update_period=0.0,
            height=240,
            width=320,
            data_types=["rgb"],
            spawn=cam_spawn,
            offset=CameraCfg.OffsetCfg(
                pos=cam_pos,
                rot=cam_rot,
                convention="ros",
            ),
        ))
        cameras_dict[cam_name] = cam

    camera_top = cameras_dict["cam_top"]
    camera_left = cameras_dict["cam_left"]
    camera_right = cameras_dict["cam_right"]

    return {
        "robot": robot,
        "rigid_cable": rigid_cable,
        "camera_top": camera_top,
        "camera_left": camera_left,
        "camera_right": camera_right,
    }


# ---------- Simulation Loop ----------


def run_simulator(sim: sim_utils.SimulationContext, entities: dict):
    """Main simulation loop with gripper cycling."""
    sim_dt = sim.get_physics_dt()
    robot = entities["robot"]
    rigid_cable = entities["rigid_cable"]
    cameras = [entities["camera_top"], entities["camera_left"], entities["camera_right"]]

    count = 0
    gripper_is_open = False
    GRIPPER_CYCLE_STEPS = 300  # Toggle every ~3s at dt=0.01

    while simulation_app.is_running():
        # Toggle gripper periodically
        if count % GRIPPER_CYCLE_STEPS == 0:
            gripper_is_open = not gripper_is_open
            grip_val = 0.025 if gripper_is_open else 0.003

            joint_pos_target = robot.data.default_joint_pos.clone()
            # Set gripper finger positions
            for jname in robot.data.joint_names:
                if "hande" in jname and "finger" in jname:
                    idx = robot.data.joint_names.index(jname)
                    joint_pos_target[:, idx] = grip_val

            robot.set_joint_position_target(joint_pos_target)
            robot.write_data_to_sim()

            state = "OPEN" if gripper_is_open else "CLOSED"
            print(f"[Step {count}] Gripper -> {state}")

        # Step simulation
        sim.step()
        count += 1

        # Update entity buffers
        robot.update(sim_dt)
        rigid_cable.update(sim_dt)
        for cam in cameras:
            cam.update(dt=sim_dt)

        # Print camera debug info periodically
        if count % 500 == 1:
            for cam in cameras:
                if "rgb" in cam.data.output:
                    print(f"  {cam.cfg.prim_path}: rgb shape = {cam.data.output['rgb'].shape}")
                # Print world pose of camera (in world convention: +X forward, +Z up)
                print(f"  {cam.cfg.prim_path}: pos_w = {cam.data.pos_w}, quat_w(world) = {cam.data.quat_w_world}")


# ---------- Main ----------


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Camera positioned to see full table scene
    sim.set_camera_view(eye=[1.5, 1.0, 1.2], target=[0.0, 0.0, 0.4])

    # Design scene
    print("[INFO] Building scene...")
    entities = design_scene()

    # Start simulation
    sim.reset()

    # Debug: print articulation body names to find correct camera parent link
    robot = entities["robot"]
    print(f"[DEBUG] Robot body names: {robot.data.body_names}")
    print(f"[DEBUG] Robot joint names: {robot.data.joint_names}")

    print("[INFO] Scene ready. Physics running.")
    print("[INFO] Gripper will cycle open/closed every ~3 seconds.")

    run_simulator(sim, entities)


if __name__ == "__main__":
    main()
    simulation_app.close()
