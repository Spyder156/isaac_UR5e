from dataclasses import field
from typing import Any


class RslRlSystemDynamicsCfg:
    def __init__(
        self,
        ensemble_size: int = 1,
        history_horizon: int = 32,
        architecture_config: dict | None = None,
        freeze_auxiliary: bool = False,
    ):
        self.ensemble_size = ensemble_size
        self.history_horizon = history_horizon
        self.architecture_config = architecture_config or {
            "type": "rnn",
            "rnn_type": "gru",
            "rnn_num_layers": 2,
            "rnn_hidden_size": 128,
            "state_mean_shape": [64],
            "state_logstd_shape": [64],
        }
        self.freeze_auxiliary = freeze_auxiliary

    def to_dict(self) -> dict:
        return {
            "ensemble_size": self.ensemble_size,
            "history_horizon": self.history_horizon,
            "architecture_config": self.architecture_config,
            "freeze_auxiliary": self.freeze_auxiliary,
        }


class RslRlNormalizerCfg:
    def __init__(self, mean: list | None = None, std: list | None = None):
        self.mean = mean or []
        self.std = std or []

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}


class RslRlMbrlImaginationCfg:
    def __init__(
        self,
        num_envs: int = 0,
        num_steps_per_env: int = 0,
        max_episode_length: int = 0,
        command_resample_interval_range: list | None = None,
        uncertainty_penalty_weight: float = 0.0,
        state_normalizer: RslRlNormalizerCfg | None = None,
        action_normalizer: RslRlNormalizerCfg | None = None,
    ):
        self.num_envs = num_envs
        self.num_steps_per_env = num_steps_per_env
        self.max_episode_length = max_episode_length
        self.command_resample_interval_range = command_resample_interval_range
        self.uncertainty_penalty_weight = uncertainty_penalty_weight
        self.state_normalizer = state_normalizer or RslRlNormalizerCfg()
        self.action_normalizer = action_normalizer or RslRlNormalizerCfg()

    def to_dict(self) -> dict:
        return {
            "num_envs": self.num_envs,
            "num_steps_per_env": self.num_steps_per_env,
            "max_episode_length": self.max_episode_length,
            "command_resample_interval_range": self.command_resample_interval_range,
            "uncertainty_penalty_weight": self.uncertainty_penalty_weight,
            "state_normalizer": self.state_normalizer.to_dict(),
            "action_normalizer": self.action_normalizer.to_dict(),
        }


class RslRlMbrlPpoAlgorithmCfg:
    def __init__(
        self,
        class_name: str = "MBPOPPO",
        value_loss_coef: float = 1.0,
        use_clipped_value_loss: bool = True,
        clip_param: float = 0.2,
        entropy_coef: float = 0.005,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        policy_learning_rate: float = 3e-4,  # Lower LR for stability
        system_dynamics_learning_rate: float = 1e-3,
        system_dynamics_weight_decay: float = 0.0,
        schedule: str = "fixed",  # Fixed schedule - no adaptive LR crashes
        gamma: float = 0.99,
        lam: float = 0.95,
        desired_kl: float = 0.02,  # Increased for less aggressive clipping
        max_grad_norm: float = 1.0,
        system_dynamics_forecast_horizon: int = 8,
        system_dynamics_loss_weights: dict | None = None,
        system_dynamics_num_mini_batches: int = 10,
        system_dynamics_mini_batch_size: int = 2000,
        system_dynamics_replay_buffer_size: int = 500,
        system_dynamics_num_eval_trajectories: int = 50,
        system_dynamics_len_eval_trajectory: int = 200,
        system_dynamics_eval_traj_noise_scale: list | None = None,
    ):
        self.class_name = class_name
        self.value_loss_coef = value_loss_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.policy_learning_rate = policy_learning_rate
        self.system_dynamics_learning_rate = system_dynamics_learning_rate
        self.system_dynamics_weight_decay = system_dynamics_weight_decay
        self.schedule = schedule
        self.gamma = gamma
        self.lam = lam
        self.desired_kl = desired_kl
        self.max_grad_norm = max_grad_norm
        self.system_dynamics_forecast_horizon = system_dynamics_forecast_horizon
        self.system_dynamics_loss_weights = system_dynamics_loss_weights or {
            "state": 1.0, 
            "sequence": 1.0, 
            "bound": 1.0, 
            "kl": 0.1,
            "extension": 1.0,
            "contact": 1.0,
            "termination": 1.0,
        }
        self.system_dynamics_num_mini_batches = system_dynamics_num_mini_batches
        self.system_dynamics_mini_batch_size = system_dynamics_mini_batch_size
        self.system_dynamics_replay_buffer_size = system_dynamics_replay_buffer_size
        self.system_dynamics_num_eval_trajectories = system_dynamics_num_eval_trajectories
        self.system_dynamics_len_eval_trajectory = system_dynamics_len_eval_trajectory
        self.system_dynamics_eval_traj_noise_scale = system_dynamics_eval_traj_noise_scale or [0.1, 0.2, 0.4]

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class UR5eReachRWMRunnerCfg:
    def __init__(self):
        self.class_name = "MBPOOnPolicyRunner"
        self.experiment_name = "ur5e_reach"
        self.run_name = "rwm"
        self.seed = 42
        
        self.num_steps_per_env = 24
        self.max_iterations = 2000
        self.save_interval = 100
        self.empirical_normalization = False
        self.clip_actions = 1.0
        self.device = "cuda:0"
        
        # Observation groups for RSL-RL
        # Maps observation sets to observation group names from environment
        # 'policy' observations are used for actor, 'critic' uses same by default
        self.obs_groups = {
            "policy": ["policy"],  # Use 'policy' obs group from env for actor
            "critic": ["policy"],  # Use same obs for critic (can also use 'system_state')
        }
        
        # Resume settings
        self.resume = False
        self.load_run = None
        self.load_checkpoint = None
        
        # Policy config (for PPO)
        self.policy = {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 256, 256],
            "critic_hidden_dims": [256, 256, 256],
            "activation": "elu",
        }
        
        # System dynamics model
        self.system_dynamics = RslRlSystemDynamicsCfg()
        
        # Imagination config (disabled for pretraining)
        self.imagination = RslRlMbrlImaginationCfg(
            num_envs=0,
            num_steps_per_env=0,
            max_episode_length=0,
            state_normalizer=RslRlNormalizerCfg(
                mean=[0.0] * 18,
                std=[1.0] * 18,
            ),
            action_normalizer=RslRlNormalizerCfg(
                mean=[0.0] * 6,
                std=[1.0] * 6,
            ),
        )
        
        # Algorithm config
        self.algorithm = RslRlMbrlPpoAlgorithmCfg()
        
        # Dynamics loading
        self.load_system_dynamics = False
        self.system_dynamics_load_path = None
        self.system_dynamics_warmup_iterations = 0
        self.system_dynamics_num_visualizations = 4
        
        # State dimension labels for plotting
        self.system_dynamics_state_idx_dict = {
            r"$q$ [rad]": [0, 1, 2, 3, 4, 5],
            r"$\dot{q}$ [rad/s]": [6, 7, 8, 9, 10, 11],
            r"$ee$ [m]": [12, 13, 14],
            r"$cube$ [m]": [15, 16, 17],
        }
        
        # PCA buffer
        self.pca_obs_buf_size = 10000

    def to_dict(self) -> dict:
        result = {}
        for k, v in self.__dict__.items():
            if hasattr(v, 'to_dict'):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result


class UR5eReachFinetuneRunnerCfg(UR5eReachRWMRunnerCfg):
    def __init__(self):
        super().__init__()
        self.run_name = "finetune"
        self.resume = True
        self.load_system_dynamics = True
        self.system_dynamics_warmup_iterations = 200
        
        # Enable imagination
        self.imagination = RslRlMbrlImaginationCfg(
            num_envs=4096,
            num_steps_per_env=24,
            max_episode_length=200,
            state_normalizer=RslRlNormalizerCfg(mean=[0.0] * 18, std=[1.0] * 18),
            action_normalizer=RslRlNormalizerCfg(mean=[0.0] * 6, std=[1.0] * 6),
        )


class UR5eReachVisualizeRunnerCfg(UR5eReachRWMRunnerCfg):
    def __init__(self):
        super().__init__()
        self.run_name = "visualize"
        self.resume = True
        self.load_system_dynamics = True
