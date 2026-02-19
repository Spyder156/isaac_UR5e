"""VisualActorCritic: ActorCritic with frozen DINOv2 encoder for image observations.

The env emits three observation groups:
  - "policy": flat proprioception (N, proprio_dim)
  - "policy_image": flattened RGB from multiple cameras (N, num_cameras * H * W * 3)
  - "camera_poses": flattened camera poses (N, num_cameras * 6) — pos(3) + forward(3) per camera

This module:
  1. Reshapes flat images → per-camera tensors
  2. Resizes to DINOv2 input resolution and normalizes with ImageNet stats
  3. Runs through frozen DINOv2 ViT-S/14 → 384D CLS token per camera
  4. Concatenates each camera's CLS token with its 6D pose → projects to 128D
  5. Fuses all camera tokens → 128D visual embedding
  6. Concatenates with proprioception → MLP actor/critic

Only the projection head and fusion MLP are trainable. DINOv2 backbone is frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.modules.actor_critic import ActorCritic


# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


class VisualActorCritic(ActorCritic):

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        # DINOv2 config
        num_cameras: int = 3,
        image_height: int = 64,
        image_width: int = 64,
        image_channels: int = 3,
        dino_model: str = "dinov2_vits14",
        dino_input_size: int = 98,
        dino_embedding_dim: int = 128,
        # Standard ActorCritic args
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        **kwargs,
    ):
        self.num_cameras = num_cameras
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.dino_input_size = dino_input_size
        self.dino_embedding_dim = dino_embedding_dim

        # Trick: replace flat image obs AND camera_poses with a fake tensor
        # of size dino_embedding_dim so parent __init__ computes correct MLP input size
        modified_obs = dict(obs)
        batch_size = obs["policy"].shape[0]
        device = obs["policy"].device
        modified_obs["policy_image"] = torch.zeros(batch_size, dino_embedding_dim, device=device)
        # Remove camera_poses from parent's dim computation — it's consumed by the encoder
        if "camera_poses" in modified_obs:
            del modified_obs["camera_poses"]

        # Also remove camera_poses from obs_groups for parent init
        modified_obs_groups = {}
        for key, groups in obs_groups.items():
            modified_obs_groups[key] = [g for g in groups if g != "camera_poses"]

        super().__init__(
            modified_obs,
            modified_obs_groups,
            num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            **kwargs,
        )

        # Restore original obs_groups (we need camera_poses in get_actor_obs)
        self.obs_groups = obs_groups

        # Load frozen DINOv2 backbone
        print(f"Loading DINOv2 backbone: {dino_model}...")
        self.dino = torch.hub.load("facebookresearch/dinov2", dino_model, pretrained=True)
        self.dino.requires_grad_(False)
        self.dino.eval()
        self.dino_feat_dim = self.dino.embed_dim  # 384 for ViT-S/14

        # Register ImageNet normalization buffers (move with model to device)
        self.register_buffer("img_mean", IMAGENET_MEAN.view(1, 3, 1, 1))
        self.register_buffer("img_std", IMAGENET_STD.view(1, 3, 1, 1))

        # Per-camera projection: DINOv2 features (384) + pose (6) → embedding
        cam_pose_dim = 6  # pos(3) + forward(3)
        self.cam_projection = nn.Sequential(
            nn.Linear(self.dino_feat_dim + cam_pose_dim, dino_embedding_dim),
            nn.ReLU(),
        )

        # Fuse all camera tokens: num_cameras * embedding → final embedding
        self.visual_fuse = nn.Sequential(
            nn.Linear(num_cameras * dino_embedding_dim, dino_embedding_dim),
            nn.ReLU(),
        )

        # Count trainable params
        dino_params = sum(p.numel() for p in self.dino.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"DINOv2 backbone: {dino_params:,} params (frozen)")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Visual pipeline: {num_cameras}x {image_height}x{image_width} → "
              f"resize {dino_input_size}x{dino_input_size} → DINOv2 {self.dino_feat_dim}D → "
              f"project+fuse → {dino_embedding_dim}D")

    def train(self, mode=True):
        """Override to keep DINOv2 always in eval mode."""
        super().train(mode)
        self.dino.eval()
        return self

    def _encode_images(self, obs: dict) -> torch.Tensor:
        """Process camera images through DINOv2 with pose conditioning."""
        flat_images = obs["policy_image"]  # (N, num_cameras * H * W * C)
        N = flat_images.shape[0]

        # Reshape: (N, num_cameras, H, W, C)
        images = flat_images.reshape(
            N, self.num_cameras, self.image_height, self.image_width, self.image_channels
        )
        # → (N * num_cameras, C, H, W) for batch processing
        images = images.permute(0, 1, 4, 2, 3).reshape(
            N * self.num_cameras, self.image_channels, self.image_height, self.image_width
        )

        # Resize to DINOv2 input resolution
        if self.image_height != self.dino_input_size or self.image_width != self.dino_input_size:
            images = F.interpolate(
                images, size=(self.dino_input_size, self.dino_input_size),
                mode="bilinear", align_corners=False,
            )

        # ImageNet normalize (images are already [0,1] from env)
        images = (images - self.img_mean) / self.img_std

        # Forward through frozen DINOv2
        with torch.no_grad():
            dino_features = self.dino(images)  # (N*num_cameras, 384)

        # Reshape to (N, num_cameras, 384)
        dino_features = dino_features.reshape(N, self.num_cameras, self.dino_feat_dim)

        # Get camera poses: (N, num_cameras * 6) → (N, num_cameras, 6)
        camera_poses = obs["camera_poses"]
        cam_poses = camera_poses.reshape(N, self.num_cameras, 6)

        # Concat features + poses and project: (N, num_cameras, 384+6) → (N, num_cameras, 128)
        cam_tokens = torch.cat([dino_features, cam_poses], dim=-1)
        cam_tokens = self.cam_projection(cam_tokens)

        # Fuse: (N, num_cameras * 128) → (N, 128)
        fused = cam_tokens.reshape(N, -1)
        visual_embedding = self.visual_fuse(fused)

        return visual_embedding  # (N, dino_embedding_dim)

    def get_actor_obs(self, obs: dict) -> torch.Tensor:
        """Override: encode images + concat with proprioception."""
        proprio_list = []
        for obs_group in self.obs_groups["policy"]:
            if obs_group in ("policy_image", "camera_poses"):
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
            if obs_group in ("policy_image", "camera_poses"):
                has_images = True
                continue
            critic_list.append(obs[obs_group])
        critic_obs = torch.cat(critic_list, dim=-1)

        if has_images:
            embedding = self._encode_images(obs)
            return torch.cat([critic_obs, embedding], dim=-1)
        return critic_obs
