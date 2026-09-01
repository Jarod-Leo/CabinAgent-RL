"""Evaluate whether Qwen2.5-7B-Instruct can enter direct CAR GRPO."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RolloutGateThresholds:
    min_trajectories: int = 80
    min_tool_call_parse_rate: float = 0.95
    min_executable_tool_rate: float = 0.85
    min_mixed_reward_group_ratio: float = 0.20
    min_consistent_initial_user_group_ratio: float = 1.0
    max_loop_or_max_turn_rate: float = 0.20
    min_successful_trajectories: int = 1
    expected_group_size: int = 4


@dataclass(frozen=True)
class RolloutGateResult:
    passed: bool
    trajectory_count: int
    complete_group_count: int
    tool_call_parse_rate: float
    executable_tool_rate: float
    mixed_reward_group_ratio: float
    consistent_initial_user_group_ratio: float
    loop_or_max_turn_rate: float
    successful_trajectories: int
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_rollout_gate(
    trajectories: list[dict[str, Any]],
    thresholds: RolloutGateThresholds | None = None,
) -> RolloutGateResult:
    thresholds = thresholds or RolloutGateThresholds()
    groups: dict[str, list[tuple[float, Any]]] = defaultdict(list)
    parse_scores: list[float] = []
    executable_scores: list[float] = []
    loop_or_max_turn = 0
    successful = 0
    reasons: list[str] = []

    for row in trajectories:
        metrics = row.get("metrics", {})
        metadata = row.get("metadata", {})
        group_id = metadata.get("group_id")
        if group_id is not None:
            outcome = float(row.get("reward", {}).get("outcome", metrics.get("success", 0.0)))
            groups[str(group_id)].append((outcome, metadata.get("first_user_message")))

        parse_value = metrics.get("tool_call_parse_rate", metadata.get("tool_call_parse_ok"))
        if parse_value is not None:
            parse_scores.append(float(parse_value))
        executable_scores.append(float(metrics.get("executable_tool_rate", 0.0)))
        successful += int(float(metrics.get("success", 0.0)) >= 1.0)

        failure_codes = {item.get("code") for item in row.get("failures", [])}
        termination = str(metadata.get("termination_reason", ""))
        if "F8_VERBOSE_OR_LOOP" in failure_codes or termination == "max_turns":
            loop_or_max_turn += 1

    complete_groups = [
        rows for rows in groups.values() if len(rows) == thresholds.expected_group_size
    ]
    mixed_groups = [
        rows
        for rows in complete_groups
        if max(item[0] for item in rows) - min(item[0] for item in rows) > 1e-8
    ]
    consistent_initial_user_groups = [
        rows
        for rows in complete_groups
        if rows[0][1] not in (None, "") and all(item[1] == rows[0][1] for item in rows)
    ]
    count = len(trajectories)
    parse_rate = _mean(parse_scores)
    executable_rate = _mean(executable_scores)
    mixed_ratio = len(mixed_groups) / len(complete_groups) if complete_groups else 0.0
    consistent_initial_user_ratio = (
        len(consistent_initial_user_groups) / len(complete_groups) if complete_groups else 0.0
    )
    loop_rate = loop_or_max_turn / count if count else 1.0

    if count < thresholds.min_trajectories:
        reasons.append(f"trajectory_count {count} < {thresholds.min_trajectories}")
    if len(parse_scores) != count:
        reasons.append("tool_call_parse_rate is missing from one or more trajectories")
    elif parse_rate < thresholds.min_tool_call_parse_rate:
        reasons.append(
            f"tool_call_parse_rate {parse_rate:.4f} < {thresholds.min_tool_call_parse_rate:.4f}"
        )
    if executable_rate < thresholds.min_executable_tool_rate:
        reasons.append(
            f"executable_tool_rate {executable_rate:.4f} < {thresholds.min_executable_tool_rate:.4f}"
        )
    if not complete_groups:
        reasons.append("no complete rollout groups were found")
    elif mixed_ratio < thresholds.min_mixed_reward_group_ratio:
        reasons.append(
            f"mixed_reward_group_ratio {mixed_ratio:.4f} < {thresholds.min_mixed_reward_group_ratio:.4f}"
        )
    if complete_groups and (
        consistent_initial_user_ratio < thresholds.min_consistent_initial_user_group_ratio
    ):
        reasons.append(
            "consistent_initial_user_group_ratio "
            f"{consistent_initial_user_ratio:.4f} < "
            f"{thresholds.min_consistent_initial_user_group_ratio:.4f}"
        )
    if loop_rate > thresholds.max_loop_or_max_turn_rate:
        reasons.append(
            f"loop_or_max_turn_rate {loop_rate:.4f} > {thresholds.max_loop_or_max_turn_rate:.4f}"
        )
    if successful < thresholds.min_successful_trajectories:
        reasons.append(
            f"successful_trajectories {successful} < {thresholds.min_successful_trajectories}"
        )

    return RolloutGateResult(
        passed=not reasons,
        trajectory_count=count,
        complete_group_count=len(complete_groups),
        tool_call_parse_rate=round(parse_rate, 6),
        executable_tool_rate=round(executable_rate, 6),
        mixed_reward_group_ratio=round(mixed_ratio, 6),
        consistent_initial_user_group_ratio=round(consistent_initial_user_ratio, 6),
        loop_or_max_turn_rate=round(loop_rate, 6),
        successful_trajectories=successful,
        reasons=reasons,
    )
