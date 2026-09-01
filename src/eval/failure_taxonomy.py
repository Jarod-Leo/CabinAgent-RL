"""Failure taxonomy shared by metrics and reports."""

from __future__ import annotations

from collections import Counter
from typing import Any


FAILURE_TYPES: dict[str, str] = {
    "F1_TOOL_NAME_ERROR": "Tool name is missing, wrong, or unsupported.",
    "F2_ARGUMENT_ERROR": "Tool arguments are missing, malformed, or wrong.",
    "F3_STATE_TRACKING_ERROR": "Multi-turn state or slot values are inconsistent.",
    "F4_MISSING_CLARIFICATION": "Ambiguous request should have been clarified.",
    "F5_CAPABILITY_HALLUCINATION": "Model invents an unavailable tool, capability, or result.",
    "F6_SAFETY_BOUNDARY_ERROR": "Model crosses a safety or capability boundary.",
    "F7_PLANNING_ORDER_ERROR": "Tool sequence or task plan order is wrong.",
    "F8_VERBOSE_OR_LOOP": "Response is unnecessarily long, repetitive, or loops.",
}


def failure(code: str, detail: str) -> dict[str, str]:
    if code not in FAILURE_TYPES:
        raise KeyError(f"Unknown failure code: {code}")
    return {"code": code, "type": FAILURE_TYPES[code], "detail": detail}


def count_failures(trajectories: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for trajectory in trajectories:
        for item in trajectory.get("failures", []):
            counter[item.get("code", "UNKNOWN")] += 1
    return counter

