import gymnasium as gym


def register_tasks():

    # Import configs here (requires Isaac Sim)
    from rwm_ur5e.configs.ur5e_reach_cfg import (
        UR5eReachEnvCfg,
        UR5eReachEnvCfg_PLAY,
        UR5eReachEnvCfg_VISUALIZE,
    )
    # Check if already registered
    if "RWM-UR5e-Reach-v0" in gym.registry:
        return
    gym.register(
        id="RWM-UR5e-Reach-v0",
        entry_point="rwm_ur5e.envs.ur5e_reach_env:UR5eReachEnv",
        disable_env_checker=True,
        kwargs={
            "cfg": UR5eReachEnvCfg(),
        },
    )
    gym.register(
        id="RWM-UR5e-Reach-Play-v0",
        entry_point="rwm_ur5e.envs.ur5e_reach_env:UR5eReachEnv",
        disable_env_checker=True,
        kwargs={
            "cfg": UR5eReachEnvCfg_PLAY(),
        },
    )
    gym.register(
        id="RWM-UR5e-Reach-Visualize-v0",
        entry_point="rwm_ur5e.envs.ur5e_reach_env:UR5eReachEnv",
        disable_env_checker=True,
        kwargs={
            "cfg": UR5eReachEnvCfg_VISUALIZE(),
        },
    )
    # --- Cable Reach (vision-based) ---
    from rwm_ur5e.configs.ur5e_cable_reach_cfg import (
        UR5eCableReachEnvCfg,
        UR5eCableReachEnvCfg_PLAY,
    )
    gym.register(
        id="RWM-UR5e-CableReach-v0",
        entry_point="rwm_ur5e.envs.ur5e_cable_reach_env:UR5eCableReachEnv",
        disable_env_checker=True,
        kwargs={
            "cfg": UR5eCableReachEnvCfg(),
        },
    )
    gym.register(
        id="RWM-UR5e-CableReach-Play-v0",
        entry_point="rwm_ur5e.envs.ur5e_cable_reach_env:UR5eCableReachEnv",
        disable_env_checker=True,
        kwargs={
            "cfg": UR5eCableReachEnvCfg_PLAY(),
        },
    )

    print("[INFO] Registered RWM UR5e gym tasks")

__all__ = ["register_tasks"]
