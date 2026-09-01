"""Deterministic CAR-specific PRM-Lite scoring rules."""

from __future__ import annotations

from typing import Any

from src.rewards.reward_types import RewardBreakdown


PROCESS_REWARD_WEIGHT = 0.3
PROCESS_SCORE_MIN = -0.5
PROCESS_SCORE_MAX = 0.5

FAILURE_WEIGHTS: dict[str, float] = {
    "F1_TOOL_NAME_ERROR": -0.08,
    "F2_ARGUMENT_ERROR": -0.07,
    "F3_STATE_TRACKING_ERROR": -0.06,
    "F4_MISSING_CLARIFICATION": -0.05,
    "F5_CAPABILITY_HALLUCINATION": -0.10,
    "F6_SAFETY_BOUNDARY_ERROR": -0.12,
    "F7_PLANNING_ORDER_ERROR": -0.06,
    "F8_VERBOSE_OR_LOOP": -0.05,
}


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _event(code: str, value: float, reason: str) -> dict[str, Any]:
    return {"code": code, "value": round(float(value), 4), "reason": reason}


def build_rule_events(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Build stable rule events from normalized metrics and environment metadata."""

    metrics = trajectory.get("metrics", {})
    metadata = trajectory.get("metadata", {})
    events: list[dict[str, Any]] = []

    for failure in trajectory.get("failures", []):
        code = str(failure.get("code", ""))
        if code in FAILURE_WEIGHTS:
            events.append(_event(code, FAILURE_WEIGHTS[code], str(failure.get("detail") or code)))

    if float(metrics.get("executable_tool_rate", 1.0)) >= 1.0:
        events.append(_event("B_TOOL_VALID", 0.03, "All emitted tool calls were executable."))
    if float(metrics.get("state_consistency", 1.0)) >= 1.0:
        events.append(_event("B_STATE_CONSISTENT", 0.03, "Tracked state remained consistent."))
    if float(metrics.get("success", 0.0)) >= 1.0:
        events.append(_event("B_FINAL_STATE", 0.08, "The requested final state was reached."))

    boolean_events = {
        "required_reads_complete": ("B_REQUIRED_READS", 0.04, "Required reads preceded mutation."),
        "clarified_ambiguity": ("B_CLARIFICATION", 0.04, "The policy resolved an ambiguity."),
        "limit_aware": ("B_LIMIT_AWARE", 0.04, "The policy respected capability limits."),
        "grounded_arguments": ("B_GROUNDED_ARGS", 0.04, "Tool arguments were grounded in observations."),
        "recovered_from_error": ("B_RECOVERY", 0.05, "The policy recovered after a tool error."),
        "state_pollution": ("P_STATE_POLLUTION", -0.08, "Unrequested state changes remained."),
        "premature_stop": ("P_PREMATURE_STOP", -0.06, "The trajectory stopped before resolution."),
        "repeated_tool_error": ("P_REPEATED_ERROR", -0.05, "A failed tool call was repeated."),
        "policy_violation": ("P_POLICY_VIOLATION", -0.12, "An automatic policy rule was violated."),
    }
    for key, (code, weight, reason) in boolean_events.items():
        if bool(metadata.get(key, False)):
            events.append(_event(code, weight, reason))

    tool_calls = int(metadata.get("tool_call_count", len(trajectory.get("predicted_tool_calls", []))))
    if tool_calls > 8:
        events.append(
            _event("P_EXCESSIVE_TOOLS", -0.01 * min(tool_calls - 8, 5), "Too many tool calls were used.")
        )

    for raw_event in metadata.get("prm_events", []):
        if not isinstance(raw_event, dict):
            continue
        code = str(raw_event.get("code", "CUSTOM"))
        value = _clip(float(raw_event.get("value", 0.0)), -0.15, 0.15)
        events.append(_event(code, value, str(raw_event.get("reason", code))))

    return events


def score_trajectory(
    trajectory: dict[str, Any], process_reward_weight: float = PROCESS_REWARD_WEIGHT
) -> RewardBreakdown:
    metrics = trajectory.get("metrics", {})
    failure_codes = {failure.get("code") for failure in trajectory.get("failures", [])}
    events = build_rule_events(trajectory)

    outcome = float(metrics.get("success", 0.0))
    tool_validity = float(metrics.get("executable_tool_rate", 1.0))
    state_consistency = float(metrics.get("state_consistency", 1.0))
    limit_awareness = 0.0 if "F5_CAPABILITY_HALLUCINATION" in failure_codes else 1.0
    disambiguation = 0.0 if "F4_MISSING_CLARIFICATION" in failure_codes else 1.0
    response_efficiency = 0.0 if "F8_VERBOSE_OR_LOOP" in failure_codes else 1.0
    process_score = _clip(
        sum(float(event["value"]) for event in events), PROCESS_SCORE_MIN, PROCESS_SCORE_MAX
    )
    total = outcome + float(process_reward_weight) * process_score
    reasons = [str(event["reason"]) for event in events]
    if not reasons:
        reasons.append("No CAR PRM-Lite rule fired.")

    return RewardBreakdown(
        total=round(total, 4),
        outcome=round(outcome, 4),
        process_score=round(process_score, 4),
        tool_validity=round(tool_validity, 4),
        state_consistency=round(state_consistency, 4),
        limit_awareness=round(limit_awareness, 4),
        disambiguation=round(disambiguation, 4),
        response_efficiency=round(response_efficiency, 4),
        reasons=reasons,
        rule_events=events,
    )


def build_prm_lite_debug_rows(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        reward = score_trajectory(trajectory)
        rows.append(
            {
                "id": trajectory.get("id"),
                "benchmark": trajectory.get("benchmark"),
                "reward": reward.to_dict(),
                "failure_codes": [item.get("code") for item in trajectory.get("failures", [])],
            }
        )
    return rows
