"""
UR5e Reach Environment for DreamerV3

This module provides the environment factory for DreamerV3 training.
"""

import gym
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class UR5eReachWrapper(gym.Env):
    """
    Wrapper that adapts the UR5eReachEnv (gymnasium) to DreamerV3's expected interface.

    DreamerV3 expects:
    - gym.Env (not gymnasium)
    - step() returns (obs, reward, done, info)
    - obs is a dict with keys matching encoder config
    - info contains 'is_first', 'is_terminal', 'discount'
    """

    def __init__(self, headless=True, time_limit=200, action_repeat=1):
        # Import here to avoid loading Isaac Sim until needed
        from isaac_envs.reach_env import UR5eReachEnv

        self._env = UR5eReachEnv(
            headless=headless,
            max_steps=time_limit,
            action_scale=0.1,
            success_threshold=0.05,
        )
        self._action_repeat = action_repeat

        # Convert gymnasium spaces to gym spaces
        # Observation space: dict with 'state' key
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
            )
        })

        # Action space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self._done = True

    def reset(self):
        """Reset environment."""
        obs, _ = self._env.reset()
        self._done = False
        return self._convert_obs(obs, is_first=True)

    def step(self, action):
        """Execute action with optional action repeat."""
        total_reward = 0.0

        for _ in range(self._action_repeat):
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break

        self._done = done
        obs = self._convert_obs(obs, is_first=False, is_terminal=done)

        # Add required DreamerV3 info
        info["is_first"] = False
        info["is_terminal"] = done
        info["discount"] = np.array(0.0 if done else 1.0, dtype=np.float32)

        return obs, total_reward, done, info

    def _convert_obs(self, obs, is_first=False, is_terminal=False):
        """Convert observation to DreamerV3 format."""
        return {
            "state": obs["state"].astype(np.float32),
            "is_first": is_first,
            "is_terminal": is_terminal,
        }

    def close(self):
        """Cleanup."""
        self._env.close()

    @property
    def unwrapped(self):
        return self._env


class DummyUR5eReach(gym.Env):
    """
    Dummy environment for testing DreamerV3 without Isaac Sim.
    Simulates the UR5e reach task with simple dynamics.
    """

    def __init__(self, time_limit=200, action_repeat=1):
        self.time_limit = time_limit
        self._action_repeat = action_repeat
        self._step_count = 0

        # State: joint_pos(6), joint_vel(6), gripper(1), ee_pos(3), cube_pos(3) = 19D
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
            )
        })
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Internal state
        self._joint_pos = np.zeros(6, dtype=np.float32)
        self._joint_vel = np.zeros(6, dtype=np.float32)
        self._ee_pos = np.array([0.4, 0.0, 0.4], dtype=np.float32)
        self._cube_pos = np.array([0.4, 0.0, 0.025], dtype=np.float32)

    def reset(self):
        self._step_count = 0
        self._joint_pos = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)
        self._joint_vel = np.zeros(6, dtype=np.float32)
        self._ee_pos = np.array([0.4, 0.0, 0.4], dtype=np.float32)
        # Random cube position
        self._cube_pos = np.array([
            np.random.uniform(0.25, 0.50),
            np.random.uniform(-0.25, 0.25),
            0.025
        ], dtype=np.float32)
        return self._get_obs(is_first=True)

    def step(self, action):
        total_reward = 0.0
        done = False

        for _ in range(self._action_repeat):
            # Simple dynamics: action is joint position delta
            action = np.clip(action, -1.0, 1.0)
            delta = action * 0.1

            self._joint_vel = delta / 0.016  # Approximate velocity
            self._joint_pos = np.clip(
                self._joint_pos + delta,
                -2*np.pi, 2*np.pi
            )

            # Simple forward kinematics approximation
            # EE moves based on last 3 joints primarily
            self._ee_pos[0] = 0.4 + 0.2 * np.sin(self._joint_pos[0])
            self._ee_pos[1] = 0.2 * np.sin(self._joint_pos[0]) * np.cos(self._joint_pos[1])
            self._ee_pos[2] = 0.3 + 0.2 * np.cos(self._joint_pos[1] + self._joint_pos[2])

            # Reward
            distance = np.linalg.norm(self._ee_pos - self._cube_pos)
            reward = -distance
            if distance < 0.05:
                reward += 10.0
                done = True

            total_reward += reward
            self._step_count += 1

            if self._step_count >= self.time_limit:
                done = True

            if done:
                break

        obs = self._get_obs(is_first=False, is_terminal=done)
        info = {
            "distance": float(np.linalg.norm(self._ee_pos - self._cube_pos)),
            "success": float(np.linalg.norm(self._ee_pos - self._cube_pos) < 0.05),
            "is_first": False,
            "is_terminal": done,
            "discount": np.array(0.0 if done else 1.0, dtype=np.float32),
        }
        return obs, total_reward, done, info

    def _get_obs(self, is_first=False, is_terminal=False):
        state = np.concatenate([
            self._joint_pos,
            self._joint_vel,
            np.array([1.0]),  # gripper open
            self._ee_pos,
            self._cube_pos,
        ]).astype(np.float32)
        return {
            "state": state,
            "is_first": is_first,
            "is_terminal": is_terminal,
        }

    def close(self):
        pass


def make_env(config, mode="train"):
    """
    Factory function for DreamerV3.

    Args:
        config: DreamerV3 config object
        mode: "train" or "eval"

    Returns:
        Environment instance
    """
    task = config.task

    if task == "ur5e_reach":
        # Use real Isaac Sim environment
        headless = mode == "train"  # Headless for training, render for eval
        env = UR5eReachWrapper(
            headless=headless,
            time_limit=config.time_limit,
            action_repeat=config.action_repeat,
        )
    elif task == "ur5e_reach_dummy":
        # Use dummy environment for testing
        env = DummyUR5eReach(
            time_limit=config.time_limit,
            action_repeat=config.action_repeat,
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    return env
