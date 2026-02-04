"""
DreamerV3 Wrapper for Isaac Lab Vectorized Environments.

Provides:
  - IsaacLabBatchWrapper: Wraps Isaac Lab's vectorized env for batched interaction.
  - simulate_isaaclab(): Drop-in replacement for tools.simulate() that operates on
    the full batch of environments natively (no per-env wrappers needed).

The key insight: DirectRLEnv.step() auto-resets done envs BEFORE computing
observations, so obs returned with done=True is already the post-reset initial
observation. This eliminates the need for explicit per-env reset calls.
"""

import datetime
import uuid

import numpy as np
import gymnasium as gym
import torch

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import tools


class IsaacLabBatchWrapper:
    """
    Wraps an Isaac Lab vectorized environment for batched DreamerV3 interaction.

    Instead of exposing individual envs, this wrapper operates on the full batch:
      - reset_all()   -> (num_envs, obs_dim) numpy
      - step_batch()  -> obs, reward, done, terminated, truncated (all numpy)

    Each sub-env is tracked with its own UUID for replay buffer keying.
    """

    def __init__(self, env, obs_key="policy"):
        self._env = env
        self._obs_key = obs_key
        self.num_envs = env.num_envs
        self.device = str(env.device)

        # Spaces (matching DreamerV3 expectations)
        obs_dim = env.cfg.observation_space   # 19
        act_dim = env.cfg.action_space        # 6

        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
        })
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        # Per-env episode tracking
        self._env_ids = [self._make_id() for _ in range(self.num_envs)]

    @staticmethod
    def _make_id():
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{timestamp}-{uuid.uuid4().hex}"

    def reset_all(self):
        """Reset all envs. Returns (num_envs, obs_dim) numpy array."""
        obs_dict, _ = self._env.reset()
        obs_tensor = obs_dict[self._obs_key]
        return obs_tensor.cpu().numpy().astype(np.float32)

    def step_batch(self, actions_np):
        """
        Step all envs with per-env actions.

        Args:
            actions_np: (num_envs, act_dim) numpy array

        Returns:
            obs_np:        (num_envs, obs_dim) numpy float32
            reward_np:     (num_envs,) numpy float64
            done_np:       (num_envs,) numpy bool
            terminated_np: (num_envs,) numpy bool
            truncated_np:  (num_envs,) numpy bool
        """
        action_tensor = torch.tensor(
            actions_np, device=self._env.device, dtype=torch.float32
        )
        obs_dict, reward, terminated, truncated, info = self._env.step(action_tensor)

        obs_np = obs_dict[self._obs_key].cpu().numpy().astype(np.float32)
        reward_np = reward.cpu().numpy()
        terminated_np = terminated.cpu().numpy().astype(bool)
        truncated_np = truncated.cpu().numpy().astype(bool)
        done_np = terminated_np | truncated_np

        return obs_np, reward_np, done_np, terminated_np, truncated_np

    def new_episode_id(self, idx):
        """Generate a fresh UUID for env slot idx (called after episode ends)."""
        self._env_ids[idx] = self._make_id()

    def close(self):
        self._env.close()


def simulate_isaaclab(
    agent,
    wrapper,
    cache,
    directory,
    logger,
    is_eval=False,
    limit=None,
    steps=0,
    episodes=0,
    state=None,
):
    """
    DreamerV3 simulate() adapted for Isaac Lab vectorized environments.

    Mirrors the logic of tools.simulate() but operates on the full batch of
    environments in a single step call.  The Dreamer agent already handles
    arbitrary batch sizes in __call__ / _policy, so passing (N, obs_dim)
    observations works without any DreamerV3 modifications.

    Args:
        agent:     callable (obs_batch, done, agent_state) -> (action_output, new_state)
        wrapper:   IsaacLabBatchWrapper instance
        cache:     OrderedDict for episode data (same as tools.simulate)
        directory: Path to save episode .npz files
        logger:    tools.Logger instance
        is_eval:   if True, log eval metrics and trim cache
        limit:     max dataset size for erase_over_episodes
        steps:     stop after this many total env transitions (OR)
        episodes:  stop after this many completed episodes
        state:     resume state from previous call, or None

    Returns:
        (step_overshoot, episode_overshoot, done, length, obs_np, agent_state, reward_np)
        Same signature as tools.simulate() for compatibility.
    """
    num_envs = wrapper.num_envs

    # --- Initialize or unpack state ---
    if state is None:
        step, episode = 0, 0
        done = np.ones(num_envs, dtype=bool)
        length = np.zeros(num_envs, dtype=np.int32)
        obs_np = wrapper.reset_all()          # (N, obs_dim)
        agent_state = None
        reward_np = np.zeros(num_envs)
    else:
        step, episode, done, length, obs_np, agent_state, reward_np = state

    # Eval metric accumulators
    eval_scores = []
    eval_lengths = []

    # --- Main loop ---
    while (steps and step < steps) or (episodes and episode < episodes):

        # 1. Handle initial / post-done transitions.
        #    For every env that was done on the previous iteration (or at init),
        #    store the initial observation with is_first=True, reward=0, discount=1.
        if done.any():
            for i in np.where(done)[0]:
                t = {
                    "state": obs_np[i].copy(),
                    "is_first": True,
                    "is_terminal": False,
                }
                t = {k: tools.convert(v) for k, v in t.items()}
                t["reward"] = 0.0
                t["discount"] = 1.0
                tools.add_to_cache(cache, wrapper._env_ids[i], t)

        # 2. Build batched observation dict for the agent.
        obs_batch = {
            "state": obs_np.copy(),                          # (N, 19)
            "is_first": done.copy(),                         # (N,)
            "is_terminal": np.zeros(num_envs, dtype=bool),   # (N,)
        }

        # 3. Get actions from agent.
        action_output, agent_state = agent(obs_batch, done, agent_state)

        # Extract numpy actions (N, act_dim)
        if isinstance(action_output, dict):
            actions_np = action_output["action"].detach().cpu().numpy()
        else:
            actions_np = np.array(action_output)

        # 4. Step all envs at once.
        obs_np, reward_np, done, terminated_np, truncated_np = (
            wrapper.step_batch(actions_np)
        )

        length += 1
        step += num_envs

        # 5. Store per-env transitions in cache.
        for i in range(num_envs):
            o = {
                "state": obs_np[i].copy(),
                "is_first": False,
                "is_terminal": bool(terminated_np[i]),
            }
            o = {k: tools.convert(v) for k, v in o.items()}
            transition = o.copy()
            if isinstance(action_output, dict):
                transition["action"] = (
                    action_output["action"][i].detach().cpu().numpy()
                )
            else:
                transition["action"] = actions_np[i]
            transition["reward"] = float(reward_np[i])
            transition["discount"] = np.array(
                0.0 if terminated_np[i] else 1.0, dtype=np.float32
            )
            tools.add_to_cache(cache, wrapper._env_ids[i], transition)

        # 6. Handle episode completions.
        if done.any():
            indices = np.where(done)[0]
            for i in indices:
                env_id = wrapper._env_ids[i]
                tools.save_episodes(directory, {env_id: cache[env_id]})
                ep_len = len(cache[env_id]["reward"]) - 1
                score = float(np.array(cache[env_id]["reward"]).sum())

                if not is_eval:
                    step_in_dataset = tools.erase_over_episodes(cache, limit)
                    logger.scalar("dataset_size", step_in_dataset)
                    logger.scalar("train_return", score)
                    logger.scalar("train_length", ep_len)
                    logger.scalar("train_episodes", len(cache))
                    logger.write(step=logger.step)
                else:
                    eval_scores.append(score)
                    eval_lengths.append(ep_len)

                # Fresh UUID for the next episode in this slot.
                wrapper.new_episode_id(i)
                episode += 1

            length[done] = 0
            # obs_np already contains post-reset obs (Isaac Lab auto-reset).

    # --- Eval aggregation ---
    if is_eval and eval_scores:
        logger.scalar("eval_return", sum(eval_scores) / len(eval_scores))
        logger.scalar("eval_length", sum(eval_lengths) / len(eval_lengths))
        logger.scalar("eval_episodes", len(eval_scores))
        logger.write(step=logger.step)
        # Keep only last episode in cache (save memory), matching tools.simulate.
        while len(cache) > 1:
            cache.popitem(last=False)

    return (step - steps, episode - episodes, done, length, obs_np, agent_state, reward_np)
