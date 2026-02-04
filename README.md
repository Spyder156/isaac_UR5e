# RWM UR5e - Robotic World Model Training

Train a UR5e robot arm using RWM (Robotic World Model) in Isaac Sim 5.1.

## Quick Start (Docker)

```bash
# Build
docker build -t rwm-ur5e -f Dockerfile.isaac .

# Run container
xhost +local:docker
docker run -it --rm \
    --gpus all \
    --privileged \
    --network host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.Xauthority:/root/.Xauthority:ro \
    -v $(pwd)/logs:/workspace/logs \
    rwm-ur5e

# Inside container
train --num_envs 64 --max_iterations 1000      # Headless training
train --num_envs 16                            # GUI training
tensorboard --logdir /workspace/logs --host 0.0.0.0  # TensorBoard
```

## Project Structure

```
intrinsic_challenge/
├── Dockerfile.isaac              # Docker build file
├── rwm_ur5e/                     # UR5e RWM environment
│   ├── configs/                  # Environment & algorithm configs
│   ├── envs/                     # Isaac Lab environment
│   └── scripts/                  # train.py, play.py, visualize.py
├── rsl_rl_rwm/                   # RSL-RL with RWM extensions
├── IsaacLab/                     # Isaac Lab framework
└── dreamerv3/                    # DreamerV3 reference implementation
```

## The Task

**UR5e Reach Task**: Move end-effector to a randomly spawned cube.

| Component | Details |
|-----------|---------|
| **Observation** | Joint pos (6) + Joint vel (6) + EE pos (3) + Cube pos (3) = 18D |
| **Action** | Joint position deltas (6D), scaled by 0.5 |
| **Reward** | Distance reward + Progress bonus - Action penalty + Stillness bonus |
| **Success** | EE within 2cm of cube |

## Training

```bash
# Basic training
python scripts/train.py --num_envs 64 --max_iterations 1000

# With specific seed
python scripts/train.py --num_envs 64 --seed 42

# Resume from checkpoint
python scripts/train.py --resume --load_run <run_name>
```

## Evaluation

```bash
# Play trained policy
python scripts/play.py --checkpoint logs/<run>/model_1000.pt

# Visualize world model predictions
python scripts/visualize.py --checkpoint logs/<run>/model_1000.pt
```

## TensorBoard

```bash
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006
# Open http://localhost:6006
```

Key metrics:
- `Train/mean_reward` - Policy performance
- `System Dynamics/autoregressive_error` - World model accuracy
- `Loss/surrogate` - PPO policy loss

## Local Setup (without Docker)

```bash
# Requires Isaac Sim 5.1 installed
conda activate robot

# Install packages
pip install -e rsl_rl_rwm
pip install -e rwm_ur5e

# Run
./run_env.sh rwm_ur5e/scripts/train.py --num_envs 64
```

## Resources

- [RWM Paper](https://arxiv.org/abs/2402.07198)
- [Isaac Sim 5.1 Docs](https://docs.omniverse.nvidia.com/isaacsim/latest/)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
