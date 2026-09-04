import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.training.best_checkpoint import BestCheckpointController, candidate_is_better


METRIC = "val-core/car_bench/reward/mean@1"


class FakeTrainer:
    def __init__(self) -> None:
        trainer = SimpleNamespace(
            save_freq=50,
            test_freq=50,
            default_local_dir="experiments/test-best/checkpoints",
            v1=SimpleNamespace(trainer_mode="sync"),
        )
        self.config = SimpleNamespace(trainer=trainer)
        self.global_steps = 0
        self.saved_steps = []
        self.validation_score = 0.20

    def _save_checkpoint(self) -> None:
        self.saved_steps.append(self.global_steps)

    def _validate(self) -> dict[str, float]:
        return {METRIC: self.validation_score}


class BestCheckpointTests(unittest.TestCase):
    def test_strict_improvement_keeps_earlier_tie(self) -> None:
        self.assertTrue(candidate_is_better(0.2, None))
        self.assertFalse(candidate_is_better(0.2, {"step": 50, "score": 0.2}))
        self.assertTrue(candidate_is_better(0.21, {"step": 50, "score": 0.2}))

    @patch("src.training.best_checkpoint.prune_run")
    @patch("src.training.best_checkpoint.validate_checkpoint")
    @patch("src.training.best_checkpoint.write_state")
    @patch("src.training.best_checkpoint.load_state")
    def test_validate_before_conditional_save(
        self,
        load_state_mock: Mock,
        write_state_mock: Mock,
        validate_checkpoint_mock: Mock,
        prune_run_mock: Mock,
    ) -> None:
        load_state_mock.return_value = {
            "schema_version": 1,
            "metric_key": METRIC,
            "comparison": "strict_greater",
            "tie_break": "earlier",
            "baseline": None,
            "best": None,
            "pending_candidate": None,
            "history": [],
        }
        trainer = FakeTrainer()
        controller = BestCheckpointController(
            trainer,
            metric_key=METRIC,
            state_path=Path("unused-best-state.json"),
        )
        controller.install()

        trainer._validate()
        self.assertEqual(trainer.saved_steps, [])
        self.assertEqual(controller.state["baseline"], {"step": 0, "score": 0.2})

        trainer.global_steps = 50
        trainer._save_checkpoint()
        self.assertEqual(trainer.saved_steps, [])
        metrics = trainer._validate()
        self.assertEqual(trainer.saved_steps, [50])
        self.assertEqual(metrics["checkpoint-selection/selected"], 1.0)

        trainer.global_steps = 100
        trainer.validation_score = 0.20
        trainer._save_checkpoint()
        trainer._validate()
        self.assertEqual(trainer.saved_steps, [50])

        trainer.global_steps = 150
        trainer.validation_score = 0.25
        trainer._save_checkpoint()
        trainer._validate()
        self.assertEqual(trainer.saved_steps, [50, 150])
        self.assertEqual(controller.state["best"], {"step": 150, "score": 0.25})
        self.assertEqual(
            [row["selected"] for row in controller.state["history"]],
            [True, False, True],
        )
        self.assertEqual(validate_checkpoint_mock.call_count, 2)
        self.assertEqual(prune_run_mock.call_count, 2)
        self.assertGreaterEqual(write_state_mock.call_count, 6)


if __name__ == "__main__":
    unittest.main()
