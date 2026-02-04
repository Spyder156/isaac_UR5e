"""
UR5e Robot Viewer with Cameras - Isaac Sim 5.1

Demonstrates a UR5e arm with Hand-E style gripper performing:
1. A choreographed dance sequence (2 loops)
2. A continuous IK-based reach toward a cube

Usage:
    ./run_env.sh view_robot.py
"""

import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

print("Starting Isaac Sim (this takes ~10 seconds)...")

from isaacsim import SimulationApp

app = SimulationApp({
    "headless": False,
    "width": 1920,
    "height": 1080,
    "anti_aliasing": 1,
})

print("Isaac Sim started. Setting up scene...")

from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane, DynamicCuboid
from isaacsim.sensors.camera import Camera
import omni.kit.commands
import numpy as np
from pxr import UsdLux, UsdGeom, UsdPhysics, Gf, PhysxSchema
from PIL import Image

# WORLD & PHYSICS
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0)
stage = world.stage

physics_context = world.get_physics_context()
physics_context.set_solver_type("TGS")
physics_context.set_gravity(-4.0)

physx_scene_prim = stage.GetPrimAtPath("/physicsScene")
if physx_scene_prim.IsValid():
    physx_api = PhysxSchema.PhysxSceneAPI.Apply(physx_scene_prim)
    physx_api.GetEnableCCDAttr().Set(True)
    physx_api.GetEnableStabilizationAttr().Set(True)
    physx_api.GetBounceThresholdAttr().Set(0.5)
    physx_api.GetFrictionOffsetThresholdAttr().Set(0.04)
    physx_api.GetGpuCollisionStackSizeAttr().Set(64 * 1024 * 1024)
    print("PhysX scene configured.")

# LIGHTS
light = UsdLux.DistantLight.Define(stage, "/World/Light")
light.GetIntensityAttr().Set(3000)
UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))

fill_light = UsdLux.DistantLight.Define(stage, "/World/FillLight")
fill_light.GetIntensityAttr().Set(1500)
UsdGeom.Xformable(fill_light).AddRotateXYZOp().Set(Gf.Vec3f(-30, -45, 0))

# GROUND & PEDESTAL
world.scene.add(GroundPlane(
    prim_path="/World/GroundPlane", name="ground_plane",
    size=10.0, color=np.array([0.3, 0.3, 0.3]),
))

pedestal_path = "/World/Pedestal"
UsdGeom.Xform.Define(stage, pedestal_path)
pedestal_base = UsdGeom.Cylinder.Define(stage, pedestal_path + "/base")
pedestal_base.GetHeightAttr().Set(0.05)
pedestal_base.GetRadiusAttr().Set(0.15)
pedestal_base.GetAxisAttr().Set("Z")
UsdGeom.Xformable(pedestal_base).AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.025))
UsdGeom.Gprim(pedestal_base).GetDisplayColorAttr().Set([Gf.Vec3f(0.15, 0.15, 0.15)])
UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(pedestal_path + "/base"))

# ROBOT (UR5e via URDF)
urdf_path = (
    "/home/raghav/miniconda3/envs/robot/lib/python3.11/site-packages/"
    "isaacsim/exts/isaacsim.robot_motion.motion_generation/"
    "motion_policy_configs/universal_robots/ur5e/ur5e.urdf"
)
print(f"Loading UR5e from: {urdf_path}")

status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.fix_base = True
import_config.make_default_prim = False
import_config.self_collision = False
import_config.create_physics_scene = False

status, robot_prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_path,
    import_config=import_config,
    dest_path="",
    get_articulation_root=True,
)
if not status:
    print("Failed to load UR5e!")
    app.close()
    exit(1)

robot_root_path = robot_prim_path.rsplit("/", 1)[0] if "root_joint" in robot_prim_path else robot_prim_path
print(f"Robot root: {robot_root_path}")

# --- Articulation stability ---
robot_prim = stage.GetPrimAtPath(robot_root_path)
if robot_prim.IsValid():
    if not robot_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        PhysxSchema.PhysxArticulationAPI.Apply(robot_prim)
    artic_api = PhysxSchema.PhysxArticulationAPI(robot_prim)
    artic_api.GetSolverPositionIterationCountAttr().Set(128)
    artic_api.GetSolverVelocityIterationCountAttr().Set(64)
    artic_api.GetEnabledSelfCollisionsAttr().Set(False)
    artic_api.GetSleepThresholdAttr().Set(0.0)

# --- Joint drives: high stiffness + high damping for position control ---
# Using "force" drive type. Stiffness is like a spring pulling toward target,
# damping prevents oscillation. Max force kept generous.
joint_configs = {
    "shoulder_pan_joint":  {"stiffness": 1e5, "damping": 1e4, "max_force": 5000.0},
    "shoulder_lift_joint": {"stiffness": 1e5, "damping": 1e4, "max_force": 5000.0},
    "elbow_joint":         {"stiffness": 1e5, "damping": 1e4, "max_force": 5000.0},
    "wrist_1_joint":       {"stiffness": 5e4, "damping": 5e3, "max_force": 2000.0},
    "wrist_2_joint":       {"stiffness": 5e4, "damping": 5e3, "max_force": 2000.0},
    "wrist_3_joint":       {"stiffness": 5e4, "damping": 5e3, "max_force": 2000.0},
}

for joint_name, cfg in joint_configs.items():
    joint_path = f"{robot_root_path}/{joint_name}"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid():
        continue
    drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
    if not drive:
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
    drive.GetTypeAttr().Set("force")
    drive.GetStiffnessAttr().Set(cfg["stiffness"])
    drive.GetDampingAttr().Set(cfg["damping"])
    drive.GetMaxForceAttr().Set(cfg["max_force"])
    # Joint friction to suppress vibration
    pj = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
    pj.GetJointFrictionAttr().Set(1.0)
    print(f"  Configured {joint_name}")

# GRIPPER
tool0_path = robot_root_path + "/tool0"
gripper_path = None
if stage.GetPrimAtPath(tool0_path).IsValid():
    gripper_path = tool0_path + "/gripper"
    UsdGeom.Xform.Define(stage, gripper_path)

    # Mount plate
    mount = UsdGeom.Cylinder.Define(stage, gripper_path + "/mount")
    mount.GetHeightAttr().Set(0.015)
    mount.GetRadiusAttr().Set(0.04)
    mount.GetAxisAttr().Set("Z")
    UsdGeom.Xformable(mount).AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.0075))
    UsdGeom.Gprim(mount).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.1, 0.1)])

    # Body
    body = UsdGeom.Cube.Define(stage, gripper_path + "/body")
    body.GetSizeAttr().Set(1.0)
    bx = UsdGeom.Xformable(body)
    bx.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.04))
    bx.AddScaleOp().Set(Gf.Vec3f(0.07, 0.04, 0.05))
    UsdGeom.Gprim(body).GetDisplayColorAttr().Set([Gf.Vec3f(0.15, 0.15, 0.15)])

    # Fingers
    for name, x_pos, x_tip in [("left", 0.025, 0.018), ("right", -0.025, -0.018)]:
        fb = UsdGeom.Cube.Define(stage, gripper_path + f"/{name}_finger_base")
        fb.GetSizeAttr().Set(1.0)
        fbx = UsdGeom.Xformable(fb)
        fbx.AddTranslateOp().Set(Gf.Vec3f(x_pos, 0, 0.08))
        fbx.AddScaleOp().Set(Gf.Vec3f(0.012, 0.025, 0.03))
        UsdGeom.Gprim(fb).GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.2, 0.2)])

        ft = UsdGeom.Cube.Define(stage, gripper_path + f"/{name}_finger_tip")
        ft.GetSizeAttr().Set(1.0)
        ftx = UsdGeom.Xformable(ft)
        ftx.AddTranslateOp().Set(Gf.Vec3f(x_tip, 0, 0.115))
        ftx.AddScaleOp().Set(Gf.Vec3f(0.008, 0.02, 0.04))
        # Make fingertips bright so gripper cam can see them
        UsdGeom.Gprim(ft).GetDisplayColorAttr().Set([Gf.Vec3f(0.9, 0.4, 0.1)])

    print("Gripper created.")

    # Note: no CollisionAPI on gripper parts — they are visual only.
    # Applying CollisionAPI causes PhysX to invalidate the articulation
    # simulation view when set_gripper_state() modifies finger transforms.
    print("Gripper created (visual only, no collision).")

# CAMERAS - wide angle
CAM_RES = (64, 64)

# Helper to set wide FOV on a camera prim (low focal length = wide angle)
def set_wide_fov(cam_prim, focal_length_mm=10.0):
    """Set camera to wide-angle by reducing focal length."""
    if not cam_prim.IsValid():
        return
    cam_geom = UsdGeom.Camera(cam_prim)
    # Focal length in scene units (mm for typical camera)
    cam_geom.GetFocalLengthAttr().Set(focal_length_mm)
    # Increase aperture for wider FOV
    cam_geom.GetHorizontalApertureAttr().Set(20.0)
    cam_geom.GetVerticalApertureAttr().Set(20.0)
    cam_geom.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

# Camera 1: Base overview — elevated, looking down at robot + cube workspace
base_cam = Camera(
    prim_path="/World/Cameras/base_camera",
    name="base_camera",
    frequency=30,
    resolution=CAM_RES,
)
base_cam_prim = stage.GetPrimAtPath("/World/Cameras/base_camera")
if base_cam_prim.IsValid():
    xf = UsdGeom.Xformable(base_cam_prim)
    xf.ClearXformOpOrder()
    # Position: elevated, behind and to the side, looking at workspace center
    xf.AddTranslateOp().Set(Gf.Vec3f(1.0, 0.6, 0.8))
    # Rotation: camera looks down -Z by default in USD
    # We want to look toward origin (0.2, 0, 0.2) from (1.0, 0.6, 0.8)
    # Approximate: pitch down ~35deg, yaw left ~30deg
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-155, 0, 150))
set_wide_fov(base_cam_prim, focal_length_mm=12.0)

# Camera 2: Wrist camera — on wrist_3_link, looking forward past tool0
wrist_cam_path = robot_root_path + "/wrist_3_link/wrist_camera"
wrist_cam = Camera(
    prim_path=wrist_cam_path,
    name="wrist_camera",
    frequency=30,
    resolution=CAM_RES,
)
wrist_cam_prim = stage.GetPrimAtPath(wrist_cam_path)
if wrist_cam_prim.IsValid():
    xf = UsdGeom.Xformable(wrist_cam_prim)
    xf.ClearXformOpOrder()
    # Offset slightly to the side of wrist
    xf.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.05, 0.0))
    # Look along +Z of tool frame (forward from gripper)
    # USD camera looks down -Z, so we rotate to align
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 90))
set_wide_fov(wrist_cam_prim, focal_length_mm=8.0)

# Camera 3: Gripper camera — inside gripper body, looking out past fingertips
if gripper_path:
    gripper_cam_path = gripper_path + "/gripper_camera"
else:
    gripper_cam_path = "/World/Cameras/gripper_camera"
gripper_cam = Camera(
    prim_path=gripper_cam_path,
    name="gripper_camera",
    frequency=30,
    resolution=CAM_RES,
)
gripper_cam_prim = stage.GetPrimAtPath(gripper_cam_path)
if gripper_cam_prim.IsValid():
    xf = UsdGeom.Xformable(gripper_cam_prim)
    xf.ClearXformOpOrder()
    # Position between fingers, slightly back so fingertips are visible at edges
    xf.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.05))
    # Look outward (along +Z of gripper frame)
    # Rotate 90 around Y to point camera +Z along gripper +Z
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))
set_wide_fov(gripper_cam_prim, focal_length_mm=6.0)  # Extra wide for fingertip view

print(f"Cameras added ({CAM_RES[0]}x{CAM_RES[1]}, wide-angle): base, wrist, gripper")

# TARGET OBJECT: cube with rigid-body physics (for pick-and-drop)
CUBE_POS = np.array([0.4, 0.0, 0.0125])
cube = world.scene.add(DynamicCuboid(
    prim_path="/World/Cube",
    name="cube",
    position=CUBE_POS,
    size=0.025,
    color=np.array([0.2, 0.6, 1.0]),
    mass=0.1,
))

# ARTICULATION HANDLE
from isaacsim.core.prims import SingleArticulation

robot_articulation = SingleArticulation(prim_path=robot_root_path, name="ur5e")
world.scene.add(robot_articulation)

world.reset()
robot_articulation.initialize()
base_cam.initialize()
wrist_cam.initialize()
gripper_cam.initialize()

num_dof = robot_articulation.num_dof
print(f"Robot has {num_dof} DOFs: {robot_articulation.dof_names}")

# Let physics settle
print("Letting physics settle...")
for _ in range(300):
    world.step(render=False)

# DEMO WITH POSES
HOME_POSE = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])

dance_poses = [
    (HOME_POSE, True),
    (np.array([0.5, -1.0, 1.0, -1.57, -1.57, 0.0]), True),    # Reach toward cube area
    (np.array([0.5, -0.7, 0.7, -1.57, -1.57, 0.0]), True),     # Lower toward cube
    (np.array([0.5, -0.7, 0.7, -1.57, -1.57, 0.0]), False),    # Close gripper
    (np.array([0.5, -1.2, 1.5, -1.57, -1.57, 0.0]), False),    # Lift
    (np.array([-0.5, -1.0, 1.0, -1.57, -1.57, 0.0]), True),    # Move to other side
    (np.array([-0.5, -0.7, 0.7, -1.57, -1.57, 0.0]), True),    # Lower
    (HOME_POSE, True),
]

# Gripper open/close helpers
finger_prims = {}
if gripper_path:
    for part in ["left_finger_base", "left_finger_tip", "right_finger_base", "right_finger_tip"]:
        p = stage.GetPrimAtPath(gripper_path + "/" + part)
        if p.IsValid():
            finger_prims[part] = p


def set_gripper_state(is_open):
    if not finger_prims:
        return
    offsets = {
        True:  {"left_finger_base": 0.025, "right_finger_base": -0.025,
                "left_finger_tip": 0.018, "right_finger_tip": -0.018},
        False: {"left_finger_base": 0.012, "right_finger_base": -0.012,
                "left_finger_tip": 0.008, "right_finger_tip": -0.008},
    }
    z_vals = {"left_finger_base": 0.08, "right_finger_base": 0.08,
              "left_finger_tip": 0.115, "right_finger_tip": 0.115}
    for part, prim in finger_prims.items():
        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3f(offsets[is_open][part], 0, z_vals[part]))
                break

# SMOOTH JOINT INTERPOLATION
# Interpolate joint positions over multiple physics steps to avoid jitter
# from instant teleportation via set_joint_positions.

current_target = dance_poses[0][0].copy()
interpolation_target = dance_poses[0][0].copy()


def set_joint_targets(positions):
    """Set a new interpolation target. Actual positions updated each step."""
    global interpolation_target
    interpolation_target = positions.copy()


def step_interpolation(alpha=0.02):
    """Move current joints toward target by alpha fraction each physics step."""
    global current_target
    current_target = current_target + alpha * (interpolation_target - current_target)
    robot_articulation.set_joint_positions(current_target)


# CAMERA CAPTURE HELPERS
camera_output_dir = "/tmp/robot_cameras"
os.makedirs(camera_output_dir, exist_ok=True)
print(f"Camera images saved to: {camera_output_dir}")

image_counter = 0


def save_cameras(tag=""):
    """Save one frame from each camera. Returns dict of numpy arrays (RGB)."""
    global image_counter
    imgs = {}
    for cam, name in [(base_cam, "base"), (wrist_cam, "wrist"), (gripper_cam, "gripper")]:
        rgba = cam.get_rgba()
        if rgba is None or rgba.size == 0:
            continue
        rgb = rgba[:, :, :3].astype(np.uint8)
        imgs[name] = rgb
        Image.fromarray(rgb).save(
            f"{camera_output_dir}/{name}_{image_counter:06d}{tag}.png"
        )
    image_counter += 1
    return imgs


# MAIN LOOP
print("\n" + "=" * 60)
print("ROBOT LOADED — dance (x2) then continuous reach toward cube")
print("Images captured at key moments.")
print("Close the window or Ctrl-C to exit.")
print("=" * 60 + "\n")

STEPS_PER_POSE = 180  # 1.5 s at 120 Hz — enough for smooth convergence
current_pose_idx = 0
step_in_pose = 0
max_loops = 2
loop_count = 0

# Set initial targets
set_joint_targets(dance_poses[0][0])
set_gripper_state(dance_poses[0][1])
pose_just_changed = True

try:
    while app.is_running() and loop_count < max_loops:
        world.step(render=True)
        step_interpolation()
        step_in_pose += 1

        # Capture one image right after a new pose target is set and the robot
        # has had a few steps to start moving (avoid blank first frame).
        if pose_just_changed and step_in_pose == 10:
            save_cameras(tag="_start")
            pose_just_changed = False

        # Capture one image near the end of each pose (robot has converged)
        if step_in_pose == STEPS_PER_POSE - 10:
            save_cameras(tag="_end")

        if step_in_pose >= STEPS_PER_POSE:
            step_in_pose = 0
            current_pose_idx += 1
            if current_pose_idx >= len(dance_poses):
                current_pose_idx = 0
                loop_count += 1
                print(f"Loop {loop_count}/{max_loops} complete.")
                if loop_count >= max_loops:
                    break

            positions, gripper_open = dance_poses[current_pose_idx]
            set_joint_targets(positions)
            set_gripper_state(gripper_open)
            pose_just_changed = True
            print(f"  Pose {current_pose_idx}/{len(dance_poses)-1}")

    # === CONTINUOUS REACH TOWARD CUBE ===
    print("\nDance done. Entering reach mode...")
    print("Robot will continuously reach toward the cube.")
    print("Close window or Ctrl-C to exit.\n")

    from isaacsim.robot_motion.motion_generation import (
        LulaKinematicsSolver,
        ArticulationKinematicsSolver,
    )

    ik_config_dir = (
        "/home/raghav/miniconda3/envs/robot/lib/python3.11/site-packages/"
        "isaacsim/exts/isaacsim.robot_motion.motion_generation/"
        "motion_policy_configs/universal_robots/ur5e"
    )
    robot_description_path = f"{ik_config_dir}/rmpflow/ur5e_robot_description.yaml"
    urdf_path_ik = f"{ik_config_dir}/ur5e.urdf"

    ik_solver = LulaKinematicsSolver(
        robot_description_path=robot_description_path,
        urdf_path=urdf_path_ik,
    )
    artic_ik = ArticulationKinematicsSolver(
        robot_articulation=robot_articulation,
        kinematics_solver=ik_solver,
        end_effector_frame_name="tool0",
    )
    print("IK solver loaded.")

    DOWN_ORIENT = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz quat, gripper Z-down
    TOOL0_TO_TIP = 0.115  # tool0 flange → fingertip distance

    cube_prim = stage.GetPrimAtPath("/World/Cube")

    def get_cube_world_pos():
        xf = UsdGeom.Xformable(cube_prim)
        tf = xf.ComputeLocalToWorldTransform(0)
        p = tf.ExtractTranslation()
        return np.array([p[0], p[1], p[2]])

    # Return to home first
    set_joint_targets(HOME_POSE)
    set_gripper_state(True)
    for _ in range(240):
        if not app.is_running():
            break
        world.step(render=True)
        step_interpolation()

    # Continuously track the cube
    cube_pos = get_cube_world_pos()
    print(f"  Cube at: {cube_pos}")
    print("  Tracking cube position (re-solving IK every 30 steps)...")

    save_cameras(tag="_reach")

    step_count = 0
    while app.is_running():
        world.step(render=True)
        step_interpolation(alpha=0.02)
        step_count += 1

        # Re-solve IK periodically to follow the cube
        if step_count % 30 == 0:
            cube_pos = get_cube_world_pos()
            reach_target = cube_pos + np.array([0, 0, TOOL0_TO_TIP + 0.05])
            action, success = artic_ik.compute_inverse_kinematics(
                target_position=reach_target,
                target_orientation=DOWN_ORIENT,
            )
            if success:
                set_joint_targets(action.joint_positions)

except KeyboardInterrupt:
    print("\nInterrupted.")

print(f"Saved {image_counter} image sets to {camera_output_dir}")
print("Closing...")
# Force-exit to avoid Isaac Sim shutdown crash in libomni.syntheticdata.plugin.so.
# app.close() itself triggers OmniGraph plugin teardown which segfaults, so skip it.
os._exit(0)
