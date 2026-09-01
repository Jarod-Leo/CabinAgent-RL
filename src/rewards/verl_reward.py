"""veRL custom-reward adapter for completed CAR trajectories."""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.training.grpo_reward_fn import outcome_reward, prm_lite_reward


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Score the trajectory attached by the CAR agent loop.

    The agent loop must place the normalized completed trajectory in
    ``extra_info["trajectory"]``. Refusing missing context prevents silently
    optimizing response text without environment state.
    """

    del data_source, solution_str, ground_truth
    trajectory = (extra_info or {}).get("trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError("CAR reward requires extra_info['trajectory'] from the agent loop")

    mode = os.environ.get("CABIN_REWARD_MODE", "outcome")
    if mode == "outcome":
        score = outcome_reward(trajectory)
    elif mode == "outcome_plus_prm_lite":
        weight = float(os.environ.get("CABIN_PROCESS_REWARD_WEIGHT", "0.3"))
        score = prm_lite_reward(trajectory, process_reward_weight=weight)
    else:
        raise ValueError(f"Unsupported CABIN_REWARD_MODE: {mode}")

    audit_dir = os.environ.get("CABIN_REWARD_AUDIT_DIR")
    if audit_dir:
        directory = Path(audit_dir)
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "mode": mode,
            "score": score,
            "task_id": trajectory.get("task_id"),
            "success": trajectory.get("success"),
            "turn_count": len(trajectory.get("turns", [])),
        }
        with (directory / f"reward-{os.getpid()}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return score
