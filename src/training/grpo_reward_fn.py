"""Reward bridge for future GRPO integration."""

from __future__ import annotations

from typing import Any

from src.rewards.prm_lite import PROCESS_REWARD_WEIGHT, score_trajectory


def outcome_reward(trajectory: dict[str, Any]) -> float:
    """Return the environment outcome component only."""

    return float(trajectory.get("metrics", {}).get("success", 0.0))


def prm_lite_reward(
    trajectory: dict[str, Any], process_reward_weight: float = PROCESS_REWARD_WEIGHT
) -> float:
    """Return a scalar PRM-Lite reward for a completed trajectory."""

    return score_trajectory(trajectory, process_reward_weight=process_reward_weight).total
