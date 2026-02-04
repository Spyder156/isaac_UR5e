"""
Test Isaac Sim reach environment.

Usage:
    ./run_env.sh                    # Headless mode (default)
    ./run_env.sh test_env.py --gui  # With GUI visualization
"""

import os
import sys
import argparse
import numpy as np

# Accept EULA before any imports
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from envs.reach_env import ReachEnv


def main():
    parser = argparse.ArgumentParser(description="Test Isaac Sim reach environment")
    parser.add_argument("--gui", action="store_true", help="Enable GUI (uses more memory)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes")
    parser.add_argument("--steps", type=int, default=50, help="Max steps per episode")
    args = parser.parse_args()

    print("=" * 60)
    print("Testing Isaac Sim Reach Environment")
    print(f"Mode: {'GUI' if args.gui else 'Headless'}")
    print("=" * 60)

    # Create environment config
    cfg = {
        'headless': not args.gui,
        'device': 'cuda:0',
        'reach_threshold': 0.02,
        'max_steps': args.steps,
        'action_scale': 0.5,
    }

    print("\n1. Creating environment...")
    try:
        env = ReachEnv(cfg)
        print("Environment created successfully!")
    except Exception as e:
        print(f"Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Run test episodes
    print(f"\n2. Running {args.episodes} test episodes...")

    for episode in range(args.episodes):
        print(f"\n   Episode {episode + 1}/{args.episodes}")

        # Reset environment
        try:
            obs, info = env.reset()
            print(f"   Reset successful. Obs shape: {obs.shape}")
        except Exception as e:
            print(f"   Reset failed: {e}")
            continue

        episode_reward = 0
        episode_length = 0
        done = False

        # Run episode
        while not done and episode_length < cfg['max_steps']:
            action = env.action_space.sample()

            try:
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1

                if episode_length % 10 == 0:
                    distance = info.get('distance', -1)
                    print(f"     Step {episode_length}: reward={reward:.3f}, distance={distance:.3f}")

            except Exception as e:
                print(f"   Step failed at step {episode_length}: {e}")
                break

        # Episode summary
        success = info.get('success', False)
        distance = info.get('distance', -1)
        print(f"   Episode finished: reward={episode_reward:.2f}, length={episode_length}, "
              f"success={success}, final_distance={distance:.3f}")

    # Cleanup
    print("\n3. Cleaning up...")
    try:
        env.close()
        print("Environment closed successfully!")
    except Exception as e:
        print(f"Cleanup failed: {e}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
