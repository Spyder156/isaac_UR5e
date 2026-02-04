# Intrinsic Challenge - Isaac Sim + DreamerV3

RL training for UR5e reach task using Isaac Sim 5.1 and DreamerV3 (PyTorch).

## Project Structure

```
intrinsic_challenge/
├── envs/                         # Simulation environments
│   ├── base_env.py               # Base Isaac Sim environment class
│   ├── reach_env.py              # UR5e reach task (current task)
│   ├── dreamer_wrapper.py        # DreamerV3 compatibility wrapper
│   └── __init__.py
├── robots/                       # Robot definitions
│   ├── ur5e_hande.py             # UR5e + Hand-E robot controller
│   └── __init__.py
├── dreamerv3/                    # DreamerV3 PyTorch implementation
│   ├── dreamer.py                # Main training logic
│   ├── models.py                 # World model components
│   ├── networks.py               # Neural network architectures
│   ├── configs.yaml              # Training hyperparameters
│   └── ...
├── ur_description/               # UR5e robot URDF and meshes
│   ├── meshes/                   # Visual and collision meshes
│   └── urdf/                     # Robot description files
├── test_env.py                   # Test the reach environment
├── train.py                      # Training script (DreamerV3)
├── run_env.sh                    # Helper script to run with correct setup
└── README.md
```

## Quick Start

### 1. Activate Environment
```bash
conda activate robot
```

### 2. Run the Environment

**Headless mode (default, faster):**
```bash
./run_env.sh
```

**With GUI visualization:**
```bash
./run_env.sh test_env.py --gui
```
Note: GUI mode uses more GPU memory. If it crashes, use headless mode.

**Options:**
```bash
./run_env.sh test_env.py --episodes 5   # Run 5 episodes
./run_env.sh test_env.py --steps 100    # 100 steps per episode
./run_env.sh test_env.py --gui --episodes 1  # Quick GUI test
```

### 3. Run Custom Scripts
```bash
./run_env.sh train.py
./run_env.sh your_script.py
```

## The Reach Task

The current implementation is a simple reach task where the UR5e robot arm must move its end-effector to a randomly placed target.

### Observation Space (12D)
| Index | Description |
|-------|-------------|
| 0-5   | Joint positions (6 arm joints, radians) |
| 6-8   | End-effector position (x, y, z in meters) |
| 9-11  | Target position (x, y, z in meters) |

### Action Space (6D)
| Index | Description |
|-------|-------------|
| 0-5   | Joint velocities (normalized to [-1, 1]) |

### Reward
- **Dense reward:** `-distance` (negative Euclidean distance to target)
- **Success bonus:** `+10.0` when distance < 2cm
- **Episode ends:** On success or after 200 steps

## How to Modify the Robot Behavior

### Change the Task (reach_env.py)

```python
# Modify reward function in _compute_reward()
def _compute_reward(self):
    distance = np.linalg.norm(ee_pos - self.target_pos)
    reward = -distance  # Your custom reward here
    return reward, done, info

# Change observation space in _get_observations()
def _get_observations(self):
    obs = np.concatenate([
        self._robot.get_joint_positions(),
        self._robot.get_ee_position(),
        self.target_pos,
        # Add more observations here
    ])
    return obs
```

### Control the Robot (ur5e_hande.py)

```python
robot = UR5eHandE(world, cfg)

# Position control (radians)
robot.set_joint_positions(np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0]))

# Velocity control (rad/s, scaled by action_scale)
robot.set_joint_velocities(np.array([0.1, 0, 0, 0, 0, 0]))

# Get robot state
joint_pos = robot.get_joint_positions()   # 6D
joint_vel = robot.get_joint_velocities()  # 6D
ee_pos = robot.get_ee_position()          # 3D
ee_pose = robot.get_ee_pose()             # 7D (pos + quaternion)
```

### Create a New Environment

1. Copy `reach_env.py` to `my_task_env.py`
2. Modify `_setup_scene()` to add objects
3. Modify `_get_observations()` for your observation space
4. Modify `_compute_reward()` for your reward function
5. Register with gymnasium at the bottom of the file

## DreamerV3 Training

The `dreamerv3/` folder contains the DreamerV3 implementation. Key files:

| File | Purpose |
|------|---------|
| `dreamer.py` | Main training loop |
| `models.py` | World model (RSSM, encoder, decoder) |
| `networks.py` | Neural network building blocks |
| `configs.yaml` | Hyperparameters |

To train:
```bash
./run_env.sh train.py
```

## System Requirements

- **GPU:** NVIDIA GPU with CUDA 12.x support
- **Driver:** NVIDIA Driver 550+
- **OS:** Linux (Ubuntu 22.04+ recommended)
- **RAM:** 16GB+ recommended
- **Conda:** `robot` environment with Python 3.11

## Troubleshooting

### GUI doesn't open
- Check `headless` is set to `False` in config
- Ensure X display is available: `echo $DISPLAY`

### CUDA errors
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Check driver: `nvidia-smi`

### URDF mesh errors
- Ensure `ur_description/` folder exists with meshes
- The `run_env.sh` script sets `ROS_PACKAGE_PATH` automatically

### Deprecation warnings
- Use `./run_env.sh -q` to filter them
- These are from Isaac Sim internal extensions, not your code

## Resources

- [Isaac Sim 5.1 Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)
- [DreamerV3 Paper](https://arxiv.org/abs/2301.04104)
- [Universal Robots ROS2 Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)
