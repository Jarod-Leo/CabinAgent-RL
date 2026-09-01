from __future__ import annotations

import math
import unittest

from src.training.long_horizon_advantage import lata_scale, normalized_exponential_weights

try:
    import torch
except ImportError:  # The CPU-only baseline intentionally has no torch dependency.
    torch = None


class LongHorizonAdvantageTests(unittest.TestCase):
    def test_weights_have_mean_one_and_favor_early_tokens(self) -> None:
        weights = normalized_exponential_weights(8, alpha=1.05)
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0, places=12)
        self.assertGreater(weights[0], weights[-1])
        self.assertTrue(all(left > right for left, right in zip(weights, weights[1:])))

    def test_alpha_one_is_uniform(self) -> None:
        self.assertEqual(normalized_exponential_weights(4, alpha=1.0), [1.0] * 4)

    def test_lata_scale_is_inverse_square_root(self) -> None:
        self.assertAlmostEqual(lata_scale(16), 0.25)
        self.assertAlmostEqual(lata_scale(2), 1.0 / math.sqrt(2))

    def test_invalid_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalized_exponential_weights(0)
        with self.assertRaises(ValueError):
            lata_scale(0)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_tensor_estimators_preserve_mask_and_lata_scaling(self) -> None:
        from src.training.long_horizon_advantage import (
            compute_grpo_lata_outcome_advantage,
            compute_grpo_turn_discounted_outcome_advantage,
        )

        rewards = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        mask = torch.ones_like(rewards)
        groups = ["same-prompt", "same-prompt"]
        discounted, _ = compute_grpo_turn_discounted_outcome_advantage(
            rewards, mask, groups
        )
        lata, _ = compute_grpo_lata_outcome_advantage(rewards, mask, groups)

        self.assertEqual(discounted.shape, rewards.shape)
        self.assertGreater(abs(discounted[1, 0]), abs(discounted[1, -1]))
        self.assertTrue(torch.allclose(lata, discounted / math.sqrt(3), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
