"""Reward data structures for PRM-Lite."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RewardBreakdown:
    total: float
    outcome: float
    process_score: float
    tool_validity: float
    state_consistency: float
    limit_awareness: float
    disambiguation: float
    response_efficiency: float
    reasons: list[str]
    rule_events: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)
