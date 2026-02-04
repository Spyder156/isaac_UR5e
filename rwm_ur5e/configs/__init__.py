# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration exports for RWM UR5e.

Note: Environment configs (ur5e_reach_cfg) require Isaac Sim to be running.
Agent configs (rsl_rl_cfg) can be imported without Isaac Sim.
"""

# Agent configs - can be imported without Isaac Sim
from .rsl_rl_cfg import (
    UR5eReachRWMRunnerCfg,
    UR5eReachFinetuneRunnerCfg,
    UR5eReachVisualizeRunnerCfg,
    RslRlSystemDynamicsCfg,
    RslRlMbrlPpoAlgorithmCfg,
    RslRlMbrlImaginationCfg,
    RslRlNormalizerCfg,
)

# Environment configs are imported lazily (require Isaac Sim)
# Use: from rwm_ur5e.configs.ur5e_reach_cfg import UR5eReachEnvCfg

__all__ = [
    # Agent configs (no Isaac Sim required)
    "UR5eReachRWMRunnerCfg",
    "UR5eReachFinetuneRunnerCfg",
    "UR5eReachVisualizeRunnerCfg",
    "RslRlSystemDynamicsCfg",
    "RslRlMbrlPpoAlgorithmCfg",
    "RslRlMbrlImaginationCfg",
    "RslRlNormalizerCfg",
]
