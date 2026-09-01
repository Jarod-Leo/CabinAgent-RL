from __future__ import annotations

import unittest

from src.rewards.prm_lite import score_trajectory


class PrmLiteTests(unittest.TestCase):
    def test_score_is_deterministic_and_bounded(self) -> None:
        trajectory = {
            "metrics": {"success": 1.0, "executable_tool_rate": 1.0, "state_consistency": 1.0},
            "failures": [],
            "predicted_tool_calls": [{"name": "read"}],
            "metadata": {"required_reads_complete": True, "grounded_arguments": True},
        }
        first = score_trajectory(trajectory)
        second = score_trajectory(trajectory)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertGreaterEqual(first.process_score, -0.5)
        self.assertLessEqual(first.process_score, 0.5)
        self.assertAlmostEqual(first.total, first.outcome + 0.3 * first.process_score)

    def test_policy_violation_is_penalized(self) -> None:
        clean = {"metrics": {"success": 0.0}, "failures": [], "metadata": {}}
        violated = {
            "metrics": {"success": 0.0},
            "failures": [],
            "metadata": {"policy_violation": True},
        }
        self.assertLess(score_trajectory(violated).process_score, score_trajectory(clean).process_score)


if __name__ == "__main__":
    unittest.main()
