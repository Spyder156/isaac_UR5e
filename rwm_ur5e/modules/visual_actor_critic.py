"""VisualActorCritic: ActorCritic with CNN encoder for image observations.

The env emits two observation groups:
  - "policy": flat proprioception (N, proprio_dim)
  - "policy_image": flattened RGB from multiple cameras (N, num_cameras * H * W * 3)

This module reshapes the flat image tensor, runs it through a CNN encoder,
and concatenates the embedding with proprioception before the MLP actor/critic.
PPO gradients flow end-to-end through the CNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic


class VisualActorCritic(ActorCritic):

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        # CNN config
        num_cameras: int = 3,
        image_height: int = 64,
        image_width: int = 64,
        image_channels: int = 3,
        cnn_embedding_dim: int = 64,
        # Standard ActorCritic args
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        **kwargs,
    ):
        self.num_cameras = num_cameras
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.cnn_embedding_dim = cnn_embedding_dim

        # Trick: replace the flat image obs with a fake tensor of size cnn_embedding_dim
        # so the parent __init__ computes the correct MLP input size
        # (proprio_dim + cnn_embedding_dim) instead of (proprio_dim + 36864)
        modified_obs = dict(obs)
        batch_size = obs["policy"].shape[0]
        modified_obs["policy_image"] = torch.zeros(batch_size, cnn_embedding_dim, device=obs["policy"].device)

        super().__init__(
            modified_obs,
            obs_groups,
            num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            **kwargs,
        )

        # Build CNN encoder
        # Input: (N, num_cameras * channels, H, W) — stack cameras along channel dim
        total_channels = num_cameras * image_channels  # 9 for 3 RGB cameras
        self.cnn = nn.Sequential(
            nn.Conv2d(total_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # (N, 64, 1, 1)
            nn.Flatten(),             # (N, 64)
        )

        # Verify CNN output matches expected embedding dim
        with torch.no_grad():
            dummy = torch.zeros(1, total_channels, image_height, image_width)
            cnn_out = self.cnn(dummy)
            assert cnn_out.shape[-1] == cnn_embedding_dim, (
                f"CNN output dim {cnn_out.shape[-1]} != expected {cnn_embedding_dim}"
            )

        print(f"VisualActorCritic CNN: {total_channels}ch {image_height}x{image_width} -> {cnn_embedding_dim}D embedding")

    def _encode_images(self, obs: dict) -> torch.Tensor:
        """Reshape flat image obs and run through CNN."""
        flat_images = obs["policy_image"]  # (N, num_cameras * H * W * C)
        N = flat_images.shape[0]

        # Reshape: (N, num_cameras, H, W, C) -> (N, num_cameras*C, H, W)
        images = flat_images.reshape(
            N, self.num_cameras, self.image_height, self.image_width, self.image_channels
        )
        images = images.permute(0, 1, 4, 2, 3).reshape(
            N, self.num_cameras * self.image_channels, self.image_height, self.image_width
        )

        return self.cnn(images)  # (N, cnn_embedding_dim)

    def get_actor_obs(self, obs: dict) -> torch.Tensor:
        """Override: encode images + concat with proprioception."""
        proprio_list = []
        for obs_group in self.obs_groups["policy"]:
            if obs_group == "policy_image":
                continue
            proprio_list.append(obs[obs_group])
        proprio = torch.cat(proprio_list, dim=-1)

        embedding = self._encode_images(obs)
        return torch.cat([proprio, embedding], dim=-1)

    def get_critic_obs(self, obs: dict) -> torch.Tensor:
        """Override: critic also uses images."""
        critic_list = []
        has_images = False
        for obs_group in self.obs_groups["critic"]:
            if obs_group == "policy_image":
                has_images = True
                continue
            critic_list.append(obs[obs_group])
        critic_obs = torch.cat(critic_list, dim=-1)

        if has_images:
            embedding = self._encode_images(obs)
            return torch.cat([critic_obs, embedding], dim=-1)
        return critic_obs
