"""Config for UR5e Cable Plug task: grasp connector + insert into socket."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg


# ---------- Scene Constants ----------

TABLE_HEIGHT = 0.4
TABLE_SURFACE_Z = TABLE_HEIGHT + 0.025  # 0.425m

# Camera params
CAM_Z = 0.03
CAM_R = 0.10
TILT = math.radians(30)
_half = TILT / 2
_c = math.cos(_half)
_s = math.sin(_half)

CAM_WIDTH = 64
CAM_HEIGHT = 64

# Socket position (on top of PCB)
SOCKET_POS = (0.5, -0.25, TABLE_SURFACE_Z + 0.015)


@configclass
class UR5eCablePlugEnvCfg(DirectRLEnvCfg):
    # Environment
    decimation = 2
    episode_length_s = 20.0  # longer for multi-phase task
    action_scale = 0.05
    action_space = 7  # 6 UR5e joints + 1 binary gripper
    observation_space = 23  # joint_pos(6) + joint_vel(6) + ee_pos(3) + gripper(1) + last_action(7)
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=2,
        device="cuda:0",
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # --- Robot: UR5e + Robotiq Hand-E ---
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="",  # Set at runtime
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
                enabled_self_collisions=True,
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
                "hande_left_finger_joint": 0.025,  # start open
                "hande_right_finger_joint": 0.025,
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

    # --- Cable: rigid articulated chain ---
    cable_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/RigidCable",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="",  # Set at runtime
            fix_base=True,
            merge_fixed_joints=False,
            self_collision=False,
            replace_cylinders_with_capsules=True,
            joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                drive_type="force",
                target_type="none",
                gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.1,
                    damping=0.5,
                ),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.2, -0.1, TABLE_SURFACE_Z + 0.005),
        ),
        actuators={},
    )

    # --- 3 Wrist Cameras ---
    cam_top_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/wrist_3_link/cam_top",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, CAM_R, CAM_Z),
            rot=(_c, _s, 0.0, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0),
        ),
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
    )

    cam_left_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/wrist_3_link/cam_left",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(CAM_R, 0.0, CAM_Z),
            rot=(_c, 0.0, -_s, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0),
        ),
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
    )

    cam_right_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/wrist_3_link/cam_right",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-CAM_R, 0.0, CAM_Z),
            rot=(_c, 0.0, _s, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0),
        ),
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
    )

    # Camera image dimensions
    camera_width: int = CAM_WIDTH
    camera_height: int = CAM_HEIGHT
    num_cameras: int = 3

    # Task parameters
    success_threshold = 0.015  # 1.5cm for insertion
    success_bonus = 20.0
    ee_link_name = "hande_left_finger"
    wrist_link_name = "wrist_3_link"
    connector_body_name = "cable_connector"
    connector_grip_body_name = "connector_grip"

    # Table
    table_height: float = TABLE_HEIGHT
    table_surface_z: float = TABLE_SURFACE_Z

    # Socket slot
    socket_pos: tuple = SOCKET_POS
    socket_slot_width: float = 0.013  # interior width
    socket_slot_height: float = 0.005  # interior height
    socket_slot_depth: float = 0.024  # depth along -Y

    # Gripper
    gripper_open_pos: float = 0.025  # max finger opening
    gripper_close_pos: float = 0.0

    # Curriculum phase boundaries (iteration counts)
    phase_1_end: int = 1500   # reach + grasp
    phase_2_end: int = 3500   # transport
    # phase 3 runs until max_iterations

    # Grasp detection thresholds
    grasp_dist_threshold: float = 0.02  # connector_grip must be within 2cm of EE
    grasp_lift_threshold: float = 0.01  # must be lifted 1cm above initial cable Z


@configclass
class UR5eCablePlugEnvCfg_PLAY(UR5eCablePlugEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
