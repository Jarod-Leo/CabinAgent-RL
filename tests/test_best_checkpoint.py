import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.checkpoint_policy import REQUIRED_RELATIVE_FILES, audit_series
from src.training.best_checkpoint import BestCheckpointController

METRIC = "val-core/car_bench/reward/mean@1"


class FakeTrainer:
    def __init__(self, root):
        self.config = SimpleNamespace(trainer=SimpleNamespace(
            save_freq=50, test_freq=50, default_local_dir=str(root / "checkpoints"),
            v1=SimpleNamespace(trainer_mode="sync")))
        self.global_steps = 0
        self.score = .2
        self.fail_save = False
        self.fail_validation = False
        self.rollout_awake = False

    def _save_checkpoint(self):
        if self.rollout_awake:
            raise RuntimeError("Saving while rollout is awake reproduces OOM")
        root = Path(self.config.trainer.default_local_dir)
        for name in REQUIRED_RELATIVE_FILES:
            path = root / f"global_step_{self.global_steps}" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        if self.fail_save:
            raise RuntimeError("Injected interrupted save")
        (root / "latest_checkpointed_iteration.txt").write_text(str(self.global_steps))

    def _validate(self):
        self.rollout_awake = True
        if self.fail_validation:
            raise RuntimeError("Injected validation failure")
        return {METRIC: self.score}


class BestCheckpointTests(unittest.TestCase):
    def test_all_boundaries_saved_and_best_ties_keep_earlier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            t = FakeTrainer(root)
            c = BestCheckpointController(t)
            c.install()
            t._validate()
            for step, score in [(50, .2), (100, .2), (150, .3), (200, .1), (250, .3)]:
                t.global_steps, t.score, t.rollout_awake = step, score, False
                t._save_checkpoint()
                self.assertTrue((root / "checkpoints" / f"global_step_{step}").exists())
                t._validate()
            result = audit_series(root, 250)
            self.assertEqual(len(result["checkpoints"]), 5)
            self.assertEqual(c.state["best"], {"step": 150, "score": .3})
            self.assertEqual(result["marker_step"], 250)

    def test_save_failure_preserves_previous_marker_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            t = FakeTrainer(root)
            c = BestCheckpointController(t)
            c.install()
            t.global_steps = 50
            t._save_checkpoint()
            t._validate()
            t.global_steps, t.rollout_awake, t.fail_save = 100, False, True
            with self.assertRaises(RuntimeError):
                t._save_checkpoint()
            self.assertEqual(t.config.trainer.default_local_dir, str(root / "checkpoints"))
            self.assertEqual((root / "checkpoints/latest_checkpointed_iteration.txt").read_text().strip(), "50")
            self.assertFalse((root / "checkpoints/global_step_100").exists())

    def test_saved_state_survives_validation_failure_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            t = FakeTrainer(root)
            c = BestCheckpointController(t)
            c.install()
            t.global_steps, t.fail_validation = 50, True
            t._save_checkpoint()
            with self.assertRaises(RuntimeError):
                t._validate()
            resumed = FakeTrainer(root)
            resumed.global_steps = 50
            resumed_controller = BestCheckpointController(resumed)
            resumed_controller.install()
            resumed._validate()
            self.assertEqual(resumed_controller.state["best"]["step"], 50)
            resumed.score = .9
            resumed._validate()
            self.assertEqual(resumed_controller.state["best"]["score"], .2)


if __name__ == "__main__":
    unittest.main()
