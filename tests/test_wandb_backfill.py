import unittest
from unittest.mock import Mock, mock_open

from scripts.backfill_wandb_from_verl import collect_metrics, parse_metric_line


class WandbBackfillTests(unittest.TestCase):
    def test_parse_metric_line_keeps_only_numeric_metrics(self) -> None:
        parsed = parse_metric_line(
            "\x1b[36m(worker)\x1b[0m step:50 - actor/loss:-1.25e-3 - "
            "val-core/car_bench/reward/mean@1:np.float64(0.25) - "
            "training/num_turns/max:np.int64(42)"
        )
        self.assertIsNotNone(parsed)
        step, metrics = parsed
        self.assertEqual(step, 50)
        self.assertEqual(metrics["actor/loss"], -1.25e-3)
        self.assertEqual(metrics["val-core/car_bench/reward/mean@1"], 0.25)
        self.assertEqual(metrics["training/num_turns/max"], 42.0)
        self.assertEqual(metrics["training/global_step"], 50.0)

    def test_collect_metrics_merges_resumed_log_segments(self) -> None:
        first = Mock()
        first.open = mock_open(read_data="step:1 - actor/loss:0.5\nstep:2 - actor/loss:0.4\n")
        second = Mock()
        second.open = mock_open(
            read_data="step:2 - critic/rewards/mean:0.25\nstep:3 - actor/loss:0.3\n"
        )
        rows = collect_metrics([first, second])
        self.assertEqual(sorted(rows), [1, 2, 3])
        self.assertEqual(rows[2]["actor/loss"], 0.4)
        self.assertEqual(rows[2]["critic/rewards/mean"], 0.25)


if __name__ == "__main__":
    unittest.main()
