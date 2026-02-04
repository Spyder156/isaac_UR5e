import argparse
import os
import sys

RWM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RWM_ROOT)
sys.path.insert(0, os.path.dirname(RWM_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train UR5e with RWM")
parser.add_argument("--task", type=str, default="RWM-UR5e-Reach-v0", help="Task ID")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=2000, help="Training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
parser.add_argument("--load_run", type=str, default=None, help="Run to load for resume")
parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint to load")
parser.add_argument("--system_dynamics_load_path", type=str, default=None, help="Dynamics model path")
parser.add_argument("--video", action="store_true", help="Record training videos")
parser.add_argument("--video_interval", type=int, default=2000, help="Video recording interval")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps")

# Add AppLauncher args (--headless, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras if recording video
if args_cli.video:
    args_cli.enable_cameras = True

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import Isaac Lab modules (requires Isaac Sim to be running)
import gymnasium as gym
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner, MBPOOnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Register our tasks (now that Isaac Sim is running)
from rwm_ur5e.envs import register_tasks
register_tasks()

# Import environment config (requires Isaac Sim)
from rwm_ur5e.configs.ur5e_reach_cfg import UR5eReachEnvCfg

# Import agent config (no Isaac Sim required, already imported structure)
from rwm_ur5e.configs import UR5eReachRWMRunnerCfg

# Enable TF32 for faster training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # Environment config
    env_cfg = UR5eReachEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    
    # Agent config
    agent_cfg = UR5eReachRWMRunnerCfg()
    agent_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations
    
    if args_cli.resume:
        agent_cfg.resume = True
        if args_cli.load_run:
            agent_cfg.load_run = args_cli.load_run
    
    if args_cli.system_dynamics_load_path:
        agent_cfg.load_system_dynamics = True
        agent_cfg.system_dynamics_load_path = args_cli.system_dynamics_load_path
    
    # Logging directory
    log_root = os.path.join(RWM_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name)
    os.makedirs(log_root, exist_ok=True)
    
    log_dir = os.path.join(
        log_root,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{agent_cfg.run_name}"
    )
    
    print(f"[INFO] Logging to: {log_dir}")
    print(f"[INFO] Num envs: {env_cfg.scene.num_envs}")
    print(f"[INFO] Max iterations: {agent_cfg.max_iterations}")
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Video recording wrapper
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording videos to: {video_kwargs['video_folder']}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    
    agent_dict = agent_cfg.to_dict()
    
    # Create runner
    if agent_cfg.class_name == "MBPOOnPolicyRunner":
        print("[INFO] Using MBPOOnPolicyRunner (model-based)")
        runner = MBPOOnPolicyRunner(
            env,
            agent_dict,
            log_dir=log_dir,
            device="cuda:0"
        )
    else:
        print("[INFO] Using OnPolicyRunner (model-free)")
        runner = OnPolicyRunner(
            env,
            agent_dict,
            log_dir=log_dir,
            device="cuda:0"
        )
    
    if args_cli.resume and args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)
    
    print(f"[INFO] Starting training for {agent_cfg.max_iterations} iterations")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
