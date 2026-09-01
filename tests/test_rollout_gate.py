from __future__ import annotations

import unittest

from src.eval.rollout_gate import RolloutGateThresholds, evaluate_rollout_gate


def make_group(group_id: str, outcomes: list[float]) -> list[dict]:
    return [
        {
            "metrics": {
                "success": outcome,
                "tool_call_parse_rate": 1.0,
                "executable_tool_rate": 0.9,
            },
            "metadata": {
                "group_id": group_id,
                "termination_reason": "completed",
                "first_user_message": f"request-{group_id}",
            },
            "failures": [],
        }
        for outcome in outcomes
    ]


class RolloutGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = RolloutGateThresholds(min_trajectories=8, expected_group_size=4)

    def test_gate_passes_with_mixed_outcome_groups(self) -> None:
        rows = make_group("g1", [0.0, 0.0, 1.0, 0.0])
        rows += make_group("g2", [0.0, 1.0, 1.0, 0.0])
        result = evaluate_rollout_gate(rows, self.thresholds)
        self.assertTrue(result.passed, result.reasons)
        self.assertEqual(result.complete_group_count, 2)
        self.assertEqual(result.mixed_reward_group_ratio, 1.0)
        self.assertEqual(result.consistent_initial_user_group_ratio, 1.0)

    def test_gate_rejects_all_zero_groups(self) -> None:
        rows = make_group("g1", [0.0] * 4) + make_group("g2", [0.0] * 4)
        result = evaluate_rollout_gate(rows, self.thresholds)
        self.assertFalse(result.passed)
        self.assertTrue(any("mixed_reward_group_ratio" in reason for reason in result.reasons))
        self.assertTrue(any("successful_trajectories" in reason for reason in result.reasons))

    def test_gate_rejects_missing_group_and_parse_metadata(self) -> None:
        rows = [{"metrics": {"success": 1.0, "executable_tool_rate": 1.0}} for _ in range(8)]
        result = evaluate_rollout_gate(rows, self.thresholds)
        self.assertFalse(result.passed)
        self.assertIn("no complete rollout groups were found", result.reasons)

    def test_gate_rejects_inconsistent_initial_user_message(self) -> None:
        rows = make_group("g1", [0.0, 0.0, 1.0, 0.0])
        rows += make_group("g2", [0.0, 1.0, 1.0, 0.0])
        rows[3]["metadata"]["first_user_message"] = "different request"
        result = evaluate_rollout_gate(rows, self.thresholds)
        self.assertFalse(result.passed)
        self.assertEqual(result.consistent_initial_user_group_ratio, 0.5)
        self.assertTrue(
            any("consistent_initial_user_group_ratio" in reason for reason in result.reasons)
        )


if __name__ == "__main__":
    unittest.main()
